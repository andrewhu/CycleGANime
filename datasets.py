
import torch
import glob
import os.path
import random
import torchvision.transforms as transforms
import cv2
from PIL import Image


class UnpairedDataset(torch.utils.data.Dataset):
    """Unpaired (or unaligned) dataset"""
    def __init__(self, data_dir, phase='test', im_size=256, crop_size=256, grayscale_A=False, seed=420):
        self.phase = phase
        # Load image paths
        self.A_paths = glob.glob(os.path.join(data_dir, phase+'A', '*.jpg'))
        self.A_size = len(self.A_paths)
        self.transform_A = get_transforms(grayscale=grayscale_A) # Get image transforms

        if phase == 'train':
            self.B_paths = glob.glob(os.path.join(data_dir, phase+'B', '*.jpg'))
            self.B_size = len(self.B_paths)
            self.transform_B = get_transforms(grayscale=False)

        self.image_size = im_size
        self.crop_size = crop_size

        # Seed RNG
        random.seed(seed)

    def resize_and_crop(self, im):
        """Resize and crop image"""
        w, h = im.size

        # Resize image maintaining aspect ratio
        if w > h:
            ratio = self.image_size / h
            new_w, new_h = int(ratio * w), self.image_size
        else:
            ratio = self.image_size / w
            new_w, new_h = self.image_size, int(ratio * h)
        im = im.resize((new_w, new_h), Image.BICUBIC)
        # if self.image_size != self.crop_size:
        # Crop image
        cx = random.randint(0, max(new_h-self.crop_size, 0))
        cy = random.randint(0, max(new_w-self.crop_size, 0))
        # print(f"cx: {cx}, cy: {cy}")

        im = im.crop((cy, cx, cy+self.crop_size, cx+self.crop_size))
        return im

    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.A_size]
        A_img = Image.open(A_path).convert('RGB')
        A_img = self.resize_and_crop(A_img) # Resize and crop image
        A = self.transform_A(A_img) # Do the other transforms we have

        if self.phase == 'train':
            B_path = self.B_paths[random.randint(0, self.B_size-1)] # choose random image
            B_img = Image.open(B_path).convert('RGB')
            # Resize and crop image
            B_img = self.resize_and_crop(B_img)
            B = self.transform_B(B_img)
        else:
            B = torch.zeros_like(A) # Empty B for inference

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size) if self.phase == 'train' else self.A_size


def get_transforms(grayscale):
    """Image transforms"""

    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(3))

    # Random flip
    transform_list.append(transforms.RandomHorizontalFlip())

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Normalize 
    transform_list.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))

    return transforms.Compose(transform_list)

