import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 随机翻转/旋转（保持不变）
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 获取图像的高度和宽度（忽略通道）
        x, y = image.shape[:2]

        if x != self.output_size[0] or y != self.output_size[1]:
            # 图像缩放：传入三个缩放因子 (h_scale, w_scale, 1)，保持通道数不变
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            # 标签缩放：传入两个缩放因子 (h_scale, w_scale)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # 转换格式
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label.astype(np.int64))
        return {'image': image, 'label': label}


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name + '.npz')
        data = np.load(data_path)
        image, label = data['image'], data['label']

        # 确保label是int64（CrossEntropyLoss需要）
        label = label.astype(np.int64)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
