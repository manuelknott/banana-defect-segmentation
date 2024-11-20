"""Images and masks require joint transforms w.r.t. to random operations"""

import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, image, target):
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
        image = functional.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        target = functional.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST_EXACT)
        return image, target.squeeze()


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = functional.center_crop(image, self.size)
        target = functional.center_crop(target, self.size)
        return image, target


class Lambda:
    def __init__(self, lambd):
        self.lambd = T.Lambda(lambd)

    def __call__(self, image, target):
        image = self.lambd(image)
        target = self.lambd(target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
        size = random.randint(self.min_size, self.max_size)
        image = functional.resize(image, size, antialias=True)
        target = functional.resize(target, size, interpolation=T.InterpolationMode.NEAREST, antialias=False)
        return image, target.squeeze()


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = functional.crop(image, *crop_params)
        target = functional.crop(target, *crop_params)
        return image, target.squeeze()


class RandomResizedCrop:
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3.0 / 4.0, 4.0 / 3.0),
                 ):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, target):
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
        crop_params = T.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        image = functional.resized_crop(image, *crop_params, size=self.size, antialias=True)
        target = functional.resized_crop(target, *crop_params, size=self.size, antialias=False,
                                         interpolation=T.InterpolationMode.NEAREST_EXACT)
        return image, target.squeeze()


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            target = functional.hflip(target)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = functional.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = functional.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image, target