import argparse
import logging

from fastai.vision import *
from torch.backends import cudnn

from callbacks import DumpPrediction, IterationCallback, TextAccuracy
from dataset import ImageDataset
from losses import loss_CAAC
from utils import Config, Logger, MyDataParallel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _set_random_seed(seed):
    if seed is not None:
        # ---- set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
        torch.backends.cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                        'This will slow down your training!')


def _get_dataset(ds_type, paths, is_training, config, **kwargs):
    kwargs.update({
        'img_h': config.dataset_imgH,
        'img_w': config.dataset_imgW,
        'contrastive': config.global_contrastive,
        'max_length': config.dataset_max_length,
        'case_sensitive': config.dataset_case_sensitive,
        'charset_path': config.dataset_charset_path,
        'data_aug': config.dataset_data_aug,
        'is_training': is_training,
        'multiscales': config.dataset_multiscales,
        'one_hot_y': config.dataset_one_hot_y,
    })
    datasets = [ds_type(p, **kwargs) for p in paths]
    return datasets[0]


def _get_databaunch(config):
    # An awkward way to reduce loadding data time during test
    if config.global_phase == 'test':
        config.dataset_train_roots = config.dataset_test_roots
    train_ds = _get_dataset(ImageDataset, config.dataset_train_roots, True, config)
    valid_ds = _get_dataset(ImageDataset, config.dataset_test_roots, False, config)
    data = ImageDataBunch.create(
        train_ds=train_ds,
        valid_ds=valid_ds,
        bs=config.dataset_train_batch_size,
        val_bs=config.dataset_test_batch_size,
        num_workers=config.dataset_data_loader_num_workers,
        pin_memory=config.dataset_data_loader_pin_memory).normalize(imagenet_stats)
    ar_tfm = lambda x: ((x[0], x[1]), x[1])  # auto-regression only for dtd
    data.add_tfm(ar_tfm)

    logging.info(f'{len(data.train_ds)} training items found.')
    if not data.empty_val:
        logging.info(f'{len(data.valid_ds)} valid items found.')

    return data


def _get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config).to(device)
    logging.info(model)
    logging.info(
        f'The parameters size of model is {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')

    if config.global_phase == 'test':
        strict = ifnone(config.model_strict, True)
        checkpoint_pth = torch.load(config.model_checkpoint, map_location='cpu')
        state_dict = checkpoint_pth['model']
        model.load_state_dict(state_dict, strict)
        logging.info(f'Read model from {config.model_checkpoint} for test')

    return model


def _get_learner(config, data, model, local_rank=None):
    metrics = [TextAccuracy(
        charset_path=config.dataset_charset_path,
        max_length=config.dataset_max_length + 1,
        case_sensitive=config.dataset_eval_case_sensisitves)]
    opt_type = getattr(torch.optim, config.optimizer_type)
    learner = Learner(data, model, silent=True, model_dir='.',
                      path=config.global_workdir,
                      metrics=metrics,
                      opt_func=partial(opt_type, **config.optimizer_args or dict()),
                      loss_func=loss_CAAC(temperature=config.global_temperature,
                                          contrastive_flag=config.global_contrastive,
                                          supervised_flag=config.global_supervised))

    learner.split(lambda m: children(m))

    if config.global_phase == 'train':
        learner.callback_fns += [
            partial(GradientClipping, clip=config.optimizer_clip_grad),
            partial(IterationCallback, name=config.global_name,
                    show_iters=config.training_show_iters,
                    eval_iters=config.training_eval_iters,
                    start_eval_iters=config.training_start_eval_iters,
                    save_iters=config.training_save_iters,
                    start_iters=config.training_start_iters,
                    stats_iters=config.training_stats_iters)]
    else:
        learner.callbacks += [
            DumpPrediction(learn=learner,
                           dataset='-'.join([Path(p).name for p in config.dataset_test_roots]),
                           charset_path=config.dataset_charset_path,
                           image_only=config.dataset_test_image_only)]

    learner.rank = local_rank
    if local_rank is not None:
        logging.info(f'Set model to distributed with rank {local_rank}.')
        learner.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(learner.model)
        learner.model.to(local_rank)
        learner = learner.to_distributed(local_rank)

    if torch.cuda.device_count() > 1 and local_rank is None:
        logging.info(f'Use {torch.cuda.device_count()} GPUs.')
        learner.model = MyDataParallel(learner.model)

    return learner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        # required=True,
                        # default='/home/ml/Desktop/CAAC/configs/CCAC_SLPR.yaml',
                        default='/home/ml/Desktop/CAAC/configs/CCAC_ch.yaml',
                        help='path to config file')
    parser.add_argument("--local_rank", type=int, default=None)

    args = parser.parse_args()
    config = Config(args.config)

    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    _set_random_seed(config.global_seed)
    logging.info(config)

    if args.local_rank is not None:
        logging.info(f'Init distribution training at device {args.local_rank}.')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logging.info('Construct dataset.')
    data = _get_databaunch(config)

    logging.info('Construct model.')
    model = _get_model(config)

    logging.info('Construct learner.')
    learner = _get_learner(config, data, model, args.local_rank)

    if config.global_phase == 'train':
        logging.info('Start training.')
        learner.fit_one_cycle(config.training_epochs,
                              config.optimizer_lr,
                              pct_start=config.training_pct_start,
                              wd=config.optimizer_wd)

    else:
        logging.info('Start validate')
        last_metrics = learner.validate()
        log_str = f'eval loss = {last_metrics[0]:6.3f},  ' \
                  f'ccr = {last_metrics[1]:6.5f},  cwr = {last_metrics[2]:6.5f},  ' \
                  f'ted = {last_metrics[3]:6.5f},  ned = {last_metrics[4]:6.5f},  ' \
                  f'ted/w = {last_metrics[5]:6.5f}, ' \
                  f'ned/w = {last_metrics[6]:6.5f}, ' \
                  f'correct word = {last_metrics[7]}, ' \
                  f'Total word = {last_metrics[8]}'
        logging.info(log_str)


if __name__ == '__main__':
    main()
