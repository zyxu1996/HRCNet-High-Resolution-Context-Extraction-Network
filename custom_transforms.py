import torch
import random
import numpy as np
import cv2


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)

        return {'image': image, 'label': label}


class RandomScaleCrop(object):
    def __init__(self, base_size=384, crop_size=384, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.choice([int(self.base_size * 0.5), int(self.base_size * 0.75), int(self.base_size),
                                    int(self.base_size * 1.25), int(self.base_size * 1.5)])
        w, h = img.shape[0:2]
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = cv2.copyMakeBorder(img, 0, padh, 0, padw, borderType=cv2.BORDER_DEFAULT)
            mask = cv2.copyMakeBorder(mask, 0, padh, 0, padw, borderType=cv2.BORDER_DEFAULT)
        # random crop crop_size
        w, h = img.shape[0:2]
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img[x1:x1+self.crop_size, y1:y1+self.crop_size, :]
        mask = mask[x1:x1+self.crop_size, y1:y1+self.crop_size]
        return {'image': img, 'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.int64).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask}
