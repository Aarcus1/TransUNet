import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BreastUltrasoundDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, image_size, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '_list.txt')).readlines()
        self.data_dir = base_dir
        self.image_size = image_size
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name)
        image_path = data_path + ".png"
        mask_path = data_path + "_mask.png"

        image = Image.open(image_path).convert('L')
        image = image.resize((self.image_size, self.image_size))
        image_np = np.array(image).astype(np.float32) / 255.0

        className = os.path.split(slice_name)[0]
        classMaskMultipliers = {
            "normal": 0.0,
            "benign": 1.0,
            "malignant": 2.0
        }
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)
        mask_np = classMaskMultipliers[className] * (np.array(mask).astype(np.float32) / 255.0)

        sample = {'image': image_np, 'label': mask_np}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
