import os
import numpy as np
import scipy.io as io
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import custom_transforms as tr
import torch


class potsdam(data.Dataset):
    def __init__(self, base_dir='./dataset', train=True):
        super(potsdam, self).__init__()
        self.dataset_dir = base_dir
        print(self.dataset_dir)
        self.train = train
        if train:
            "RGB"
            self.image_dir = os.path.join(self.dataset_dir, 'images/train/')
            self.label_dir = os.path.join(self.dataset_dir, 'labels/train/')
        else:
            self.image_dir = os.path.join(self.dataset_dir, 'images/test/')
            self.label_dir = os.path.join(self.dataset_dir, 'labels/test/')

        self.filename_list = os.listdir(self.image_dir)

        self.im_ids = []
        self.images = []
        self.labels = []
        for filename in self.filename_list:
            image = os.path.join(self.image_dir, filename.strip())
            label = os.path.join(self.label_dir, filename.strip()[:-4] + ".png")
            self.images.append(image)
            self.labels.append(label)
        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.make_img_gt_point_pair(index)
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        image, label = sample['image'], sample['label']
        if self.train:
            edge = torch.from_numpy(np.array(Image.fromarray(edge_contour(np.asarray(label))))).long()
            sample = {'image': image, 'label': label, 'edge': edge}
            return sample
        else:
            return sample

    def make_img_gt_point_pair(self, index):
        image = io.loadmat(self.images[index])['ALL']  # read .mat images
        label = Image.open(self.labels[index])
        label = np.array(label)
        return image, label

    def transform(self, sample):
        if self.train:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomScaleCrop(),
                tr.ToTensor()
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.ToTensor()
            ])
        return composed_transforms(sample)

    def __str__(self):
        if self.train:
            return 'Potsdam(train=True)'
        else:
            return 'Potsdam(train=False)'


def edge_contour(label, edge_width=3):
    import cv2 as cv

    _, h, w = label.shape
    label = label.squeeze()
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (edge_width, edge_width))
    edge = cv.dilate(edge, kernel)

    # randomx = random.randint(1, 100)
    # scipy.misc.imsave('./edge/{}.png'.format(randomx), label_to_RGB(edge))
    # scipy.misc.imsave('./test_label/{}.png'.format(randomx), label_to_RGB(label))
    return edge


def label_to_RGB(image):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    index = image == 0
    RGB[index] = np.array([255, 255, 255])
    index = image == 1
    RGB[index] = np.array([0, 0, 255])
    index = image == 2
    RGB[index] = np.array([0, 255, 255])
    index = image == 3
    RGB[index] = np.array([0, 255, 0])
    index = image == 4
    RGB[index] = np.array([255, 255, 0])
    index = image == 5
    RGB[index] = np.array([255, 0, 0])
    return RGB


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    Potsdam_train = potsdam(train=True)
    dataloader = DataLoader(Potsdam_train, batch_size=1, shuffle=False, num_workers=1)
    # print(dataloader)

    for ii, sample in enumerate(dataloader):
        im = sample['label'].numpy().astype(np.uint8)
        pic = sample['image'].numpy().astype(np.uint8)
        print(im.shape)
        im = np.squeeze(im, axis=0)
        pic = np.squeeze(pic, axis=0)
        print(im.shape)
        im = np.transpose(im, axes=[1, 2, 0])[:, :, 0:3]
        pic = np.transpose(pic, axes=[1, 2, 0])[:, :, 0:3]
        print(im.shape)
        im = np.squeeze(im, axis=2)
        print(im)
        im = label_to_RGB(im)
        plt.imshow(pic)
        plt.show()
        plt.imshow(im)
        plt.show()
        if ii == 10:
            break
