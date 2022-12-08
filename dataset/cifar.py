"""
set cifar dataset
"""

import os
from paddle.vision.datasets import cifar
from .transforms import BaseTransform, ListTransform
from PIL import Image
import numpy as np
import shutil


def np_convert_pil(array):
    """
    array conver image
    Args:
        array: array and dim is 1
    """
    assert len(array.shape), "dim of array should 1"
    img = Image.fromarray(array.reshape(3, 32, 32).transpose(1, 2, 0))
    return img


class CIFAR10(cifar.Cifar10):
    """
    cifar10 dataset
    """
    def __init__(self, root, mode='train'):
        super().__init__(download=True, mode=mode)
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(os.path.join(root, 'cifar-10-python.tar.gz')):
            shutil.move('~/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz', root)
        
        self.x = []
        self.y = []
        for d in self.data:
            self.x.append(d[0])
            self.y.append(d[1])

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]


class CIFAR10SSL(CIFAR10):
    """
    from Cifar10
    """

    def __init__(self, root, index, transforms, mode='train'):
        super().__init__(root, mode=mode)
        # self.x = []
        # self.y = []
        # for d in self.data:
        #     self.x.append(d[0])
        #     self.y.append(d[1])

        # self.x = np.array(self.x)
        # self.y = np.array(self.y)
        if index is not None:
            self.x = self.x[index]
            self.y = self.y[index]
        self.index = index
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, idx):
        img, label = np_convert_pil(self.x[idx]), self.y[idx]
        results = ListTransform(self.transforms)(img)
        # if self.index is not None:
        #     print(results[0][0][0][0].dtype)
            
        return results, label
        
    def __len__(self):
        return self.x.shape[0]


def x_u_split(cfg, label):
    """
    split index of dataset to labeled x and unlabeled u
    Args:
        num_labeled: num of labeled dataset
        label: list or array, label
    """
    num_labeled = cfg['num_labeled']
    num_classes = cfg['num_classes']
    assert num_labeled <= len(label), "arg num_labeled should <= num of label"
    label = np.array(label) if isinstance(label, list) else label
    label_per_class = num_labeled // num_classes
    labeled_idx = []
    unlabeled_idx = np.array(list(range(label.shape[0])))
    for c in range(num_classes):
        idx = np.where(label == c)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx