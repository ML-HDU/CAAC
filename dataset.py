import logging
import re

import cv2
import lmdb
import matplotlib.pyplot as plt
import six
from fastai.vision import *
from torchvision import transforms

from transforms import get_augmentation_pipeline
from utils import CharsetMapper, onehot


class ImageDataset(Dataset):
    "`ImageDataset` read data from LMDB database."

    def __init__(self,
                 path: PathOrStr,
                 is_training: bool = True,
                 contrastive: bool = False,
                 img_h: int = 64,
                 img_w: int = 256,
                 max_length: int = 25,
                 check_length: bool = True,
                 case_sensitive: bool = False,
                 charset_path: str = 'data/alphabet_ch_6067.txt',
                 convert_mode: str = 'RGB',
                 data_aug: bool = True,
                 multiscales: bool = True,
                 one_hot_y: bool = True,
                 return_idx: bool = False,
                 return_raw: bool = False,
                 **kwargs):
        self.path, self.name = Path(path), Path(path).name
        assert self.path.is_dir() and self.path.exists(), f"{path} is not a valid directory."
        self.convert_mode, self.check_length = convert_mode, check_length
        self.img_h, self.img_w = img_h, img_w
        self.contrastive = contrastive
        self.max_length, self.one_hot_y = max_length, one_hot_y
        self.return_idx, self.return_raw = return_idx, return_raw
        self.case_sensitive, self.is_training = case_sensitive, is_training
        self.data_aug, self.multiscales = data_aug, multiscales
        self.charset = CharsetMapper(charset_path, max_length=max_length + 1)
        self.c = self.charset.num_classes

        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))
            
        self.augmentation_severity = 1
        
        if self.is_training and self.data_aug:
            self.augment_tfs = get_augmentation_pipeline().augment_image
                
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return self.length

    def _next_image(self, index):
        next_index = random.randint(0, len(self) - 1)
        return self.get(next_index)

    def _check_image(self, x, pixels=6):
        if x.size[0] <= pixels or x.size[1] <= pixels:
            return False
        else:
            return True

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_w < self.img_h:
                if fix_h:
                    trg_h = self.img_h
                else:
                    trg_h = int(ratio * self.img_w)
                trg_w = self.img_w
            else:
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            trg_w = max(trg_w, 1)
            trg_h = max(trg_h, 1)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            
            return img

        if self.is_training:
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h / w)
            else:
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        if self.multiscales:
            return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        else:
            return cv2.resize(img, (self.img_w, self.img_h))

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
            try:
                label = str(txn.get(label_key.encode()), 'utf-8')  # label

                if self.check_length and self.max_length > 0:
                    if len(label) > self.max_length or len(label) <= 0:
                        logging.info(f'Long or short text image is found: {self.name}, {idx}, {label}, {len(label)}')
                        return self._next_image(idx)
                label = label[:self.max_length]

                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    image = PIL.Image.open(buf).convert(self.convert_mode)
                if self.is_training and not self._check_image(image):
                    logging.info(f'Invalid image is found: {self.name}, {idx}, {label}, {len(label)}')
                    return self._next_image(idx)
            except:
                import traceback
                traceback.print_exc()
                logging.info(f'Corrupted image is found: {self.name}, {idx}, {label}, {len(label)}')
                return self._next_image(idx)

            return image, label, idx

    def _process_training(self, image):
        image = np.array(image)
        if self.data_aug and not self.contrastive:
            image = self.augment_tfs(image)
            image = self.resize(np.array(image))
        elif self.data_aug and self.contrastive:
            image_1 = self.augment_tfs(image)
            image_2 = self.augment_tfs(image)

            image_1 = self.resize(np.array(image_1))
            image_2 = self.resize(np.array(image_2))

            image = [image_1, image_2]
        else:
            image = self.resize(np.array(image))
        return image

    def _process_test(self, image):
        return self.resize(np.array(image))  # TODO:move is_training to here

    def __getitem__(self, idx):
        image, text, idx_new = self.get(idx)

        if self.is_training:
            image = self._process_training(image)
        else:
            image = self._process_test(image)
        if self.return_raw:
            return image, text

        if not self.contrastive or not self.is_training:
            image = self.totensor(image)
        elif self.contrastive:
            image_1 = self.totensor(image[0])
            image_2 = self.totensor(image[1])
            image = np.stack([image_1, image_2], axis=0)

        length = tensor(len(text) + 1).to(dtype=torch.long)  # one for end token
        label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        label = tensor(label).to(dtype=torch.long)
        if self.one_hot_y: label = onehot(label, self.charset.num_classes)

        if self.return_idx:
            y = [label, length, idx_new]
        else:
            y = [label, length]
        return image, y
