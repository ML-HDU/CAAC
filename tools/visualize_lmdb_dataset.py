#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
=================================================
# @File   ：visualize_lmdb_dataset.py
# @IDE    ：PyCharm
# @Author ：Tnak
# @Date   ：2022/5/8 下午2:14
# @Desc   ：可视化lmdb格式的文字识别数据集
==================================================
"""
import random
from re import L

import matplotlib.pyplot as plt
import numpy as np
import zhconv
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import lmdb                                         # lmdb 是一种数据库，可以实现多进程访问，访问简单，而且不需要把全部文件读入内存
import six
import sys
import math
import torch
import logging
from PIL import Image
from tqdm import tqdm

from transforms import get_augmentation_pipeline


def set_random_seed(seed):
    if seed is not None:
        # ---- set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                        'This will slow down your training!')


class lmdbDataset(Dataset):

    def __init__(self, root=None):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(path=root,             # 存储数据库的目录位置 或 文件前缀
                             max_readers=16,        # 同时读取事务的最大数量
                             readonly=True,         # 禁止任何 写 操作
                             lock=False,            # 如果 False，不要做任何锁定
                             readahead=False,       # 如果为 False，LMDB将禁用OS文件系统预读机制，这可能会在数据库大于RAM时提高随机读取性能
                             meminit=False)         # 如果为 False，LMDB将不会在将缓冲区写入磁盘时对其进行零初始化，这提高了性能。

        if not self.env:
            print('cannot create lmdb from {}'.format(root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        self.transform = transforms.ToTensor()

        self.augment_tfs_ours = get_augmentation_pipeline(ours=True).augment_image
        self.augment_tfs_SeqCLR = get_augmentation_pipeline(ours=False).augment_image

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            try:
                img = Image.open(buf).convert('RGB')

                # image_1 = self.augment_tfs_ours(np.array(img))
                # image_2 = self.augment_tfs_ours(np.array(img))
                # fig = plt.figure()
                # sub = fig.add_subplot(2, 3, 1)
                # sub.imshow(np.array(img))
                # sub1 = fig.add_subplot(2, 3, 2)
                # sub1.imshow(image_1)
                # sub2 = fig.add_subplot(2, 3, 3)
                # sub2.imshow(image_2)
                #
                # image_1 = self.augment_tfs_SeqCLR(np.array(img))
                # image_2 = self.augment_tfs_SeqCLR(np.array(img))
                # sub = fig.add_subplot(2, 3, 4)
                # sub.imshow(np.array(img))
                # sub1 = fig.add_subplot(2, 3, 5)
                # sub1.imshow(image_1)
                # sub2 = fig.add_subplot(2, 3, 6)
                # sub2.imshow(image_2)
                #
                # plt.savefig(str(label) + '.svg')

                img = self.transform(img)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

        return img, label


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
      inside_code = ord(uchar)
      if inside_code == 12288:  # 全角空格直接转换
        inside_code = 32
      elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
        inside_code -= 65248
      rstring += chr(inside_code)
    return rstring


def augmentation(augment_tfs, image):
    image = np.array(image)
    image_1 = augment_tfs(image)
    image_2 = augment_tfs(image)

    return image_1, image_2


if __name__ == '__main__':
    set_random_seed(seed=42)

    benchmark = 'images/Scene_Text_dataset'
    dataset_path = r'/media/ml/DATA/Various_Datasets/Scene_Text_Recognition/recognition_datasets/Chinese_Scene_Dataset_FuDan/scene/scene_test'
    # dataset_path = r'/media/ml/DATA/Various_Datasets/Scene_Text_Recognition/recognition_datasets/Chinese_Scene_Dataset_FuDan/web/web_test'
    # dataset_path = r'/media/ml/DATA/Various_Datasets/SLPR_recognition/rectangle_test/test_LMDB_rectangle'
    dataset = lmdbDataset(root=dataset_path)
    dataloader = DataLoader(dataset, batch_size=384, shuffle=False)

    obj_texts = ['雕刻销售各形大理石花岗石墓碑竣工碑功德碑']
    all_texts = []
    texts_nums = {}
    for i, batch in enumerate(tqdm(dataloader)):
        img, label = batch

        gt_text = [zhconv.convert(strQ2B(gt.lower().replace(' ', '')), 'zh-cn') for gt in label]
        # for text in gt_text[0]:
        #     if text in all_texts:
        #         texts_nums[text] += 1
        #     else:
        #         all_texts.append(text)
        #         texts_nums[text] = 1
        # all_texts = list(set(all_texts))
        if gt_text[0] in obj_texts:
            torchvision.utils.save_image(img, benchmark + '_' + str(gt_text[0]) + '.jpg')
            print('The label of ' + benchmark + '_' + str(label[0]) + '.jpg ' 'is ' + str(gt_text[0]))
        print(i, '----', gt_text[0])

    print('The chosen texts for t-sne are', random.sample(all_texts, 10))

    # with open('./text_nums.txt', 'w', encoding='utf-8') as f:
        # for key, value in texts_nums.items():
        #     f.write(key)
        #     f.write(': ')
        #     f.write(str(value))
        #     f.write('\n')





