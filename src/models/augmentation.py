import random

import torch
import torch.nn as nn
from torchvision.transforms import v2


class SaltAndPepperNoise():
    def __init__(self, max_prob=0.0073):
        """Adds impulse noise ("salt and pepper") to the image.

        Args:
            prob (float, optional): the proportion of pixels that will be replaced. Defaults to 0.0073.
        """
        self.max_prob = max_prob

    def __call__(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input should be a tensor")

        noisy = tensor.clone()
        c, h, w = noisy.shape
        prob = random.uniform(0.0, self.max_prob)
        num_pixels = int(prob * h * w)

        for ch in range(c):
            coords_salt = [
                (random.randint(0, h - 1), random.randint(0, w - 1))
                for _ in range(num_pixels // 2)
            ]
            coords_pepper = [
                (random.randint(0, h - 1), random.randint(0, w - 1))
                for _ in range(num_pixels // 2)
            ]

            for y, x in coords_salt:
                noisy[ch, y, x] = 1.0
            for y, x in coords_pepper:
                noisy[ch, y, x] = 0.0

        return noisy


class TableAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = v2.Compose([
            v2.RandomAffine(
                degrees=15,
                scale=(1.0, 1.2),
                translate=(0.05, 0.05)
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.05
            ),
            v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            # SaltAndPepperNoise(max_prob=0.0073)
        ])
    
    def forward(self, img):
        return self.transforms(img)