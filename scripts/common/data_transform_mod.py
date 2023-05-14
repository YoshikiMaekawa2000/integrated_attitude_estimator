from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, resize, mean, std, brightness, contrast, saturation, hue, kernel_size, sigma_min, sigma_max, equalize_p, elastic_alpha, phase):
        self.mean = mean
        self.std = std
        size = (resize, resize)

        if phase == "train":
            self.img_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=(sigma_min, sigma_max)),
                transforms.RandomEqualize(p=equalize_p),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.5), ratio=(0.3, 3.3), value=1.0),
                transforms.Normalize((mean,), (std,))
            ])
        elif phase == "distort":
            self.img_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=(sigma_min, sigma_max)),
                transforms.RandomEqualize(p=equalize_p),
                transforms.ElasticTransform(alpha=elastic_alpha),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.5), ratio=(0.3, 3.3), value=1.0),
                transforms.Normalize((mean,), (std,))
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])

    def __call__(self, img_pil, roll_numpy, pitch_numpy, phase="train"):
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        
        ## roll: numpy -> tensor
        roll_numpy = roll_numpy.astype(np.float32)
        roll_tensor = torch.from_numpy(roll_numpy)

        # pitch: numpy -> tensor
        pitch_numpy = pitch_numpy.astype(np.float32)
        pitch_tensor = torch.from_numpy(pitch_numpy)
        
        #return img_tensor, logged_roll_tensor, logged_pitch_tensor
        return img_tensor, roll_tensor, pitch_tensor