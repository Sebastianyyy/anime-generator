import torch
import os
from .utils import current_directory
from torchvision.io import read_image

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = os.listdir(
            os.path.join(current_directory, self.img_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        idx += 1 # because the images starts from 1
        img_path = os.path.join(
            current_directory, self.img_dir, str(idx)+'.png') # all images are png
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image
