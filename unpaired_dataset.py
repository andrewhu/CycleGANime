
import torch
import glob
import os.path
import random
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class UnpairedDataset(torch.utils.data.Dataset):
    """Unpaired (or unaligned) dataset"""
    def __init__(self, data_dir, phase='test', im_size=256, crop_size=256):
        self.phase = phase

        # Load image paths
        self.A_paths = glob.glob(os.path.join(data_dir, self.phase+'A', '*.jpg'))
        self.A_size = len(self.A_paths)
        self.transform_A = get_transforms(grayscale=True) # Get image transforms

        if self.phase == 'train':
            self.B_paths = glob.glob(os.path.join(data_dir, self.phase+'B', '*.jpg'))
            self.B_size = len(self.B_paths)
            self.transform_B = get_transforms(grayscale=False)

        self.image_size = im_size
        self.crop_size = crop_size

    def resize_and_crop(self, im):
        """Resize and crop image"""
        
        # Resize image, maintaining aspect ratio
        w, h = im.size
        if w > h:
            ratio = self.image_size / h
            new_w, new_h = int(ratio * w), self.image_size
        else:
            ratio = self.image_size / w
            new_w, new_h = self.image_size, int(ratio * h)
        im = im.resize((new_w, new_h), Image.BICUBIC)

        # Crop image
        x = random.randint(0, max(new_h-self.crop_size, 0))
        y = random.randint(0, max(new_w-self.crop_size, 0))
        im = im.crop((y, x, y+self.crop_size, x+self.crop_size))

        return im


    def set_im_mean(self, im, mean=200, max_iters=50):
        """
        Iteratively brightens or darkens an image until it reaches a 
        chosen mean intensity value. This is so that the average pixel
        intensity of the inputs are the same, which improves quality.

        Consider this as a type of normalization
        """
        im = im.astype(np.float32)
        
        # Mean of non-white pixels
        initial_mean = im[np.where(im<254)].mean()

        # Iteratively brighten or darken non-white pixels
        if initial_mean < mean: # image is darker
            for _ in range(max_iters):
                im_mean = im[im<254].mean()
                if im_mean > mean: # If our image is bright enough
                    break
                x = abs(mean - im_mean)/255 + 0.005
                im[im<254] = im[im<254]*(1-x)+255*x # Brighten image
        else: # image is brighter
            for _ in range(max_iters):
                im_mean = im[im<254].mean()
                if im_mean < mean: # If our image is dark enough
                    break
                im[im<254] = im[im<254] * 0.995 - 255 * .005 # Darken image

        return im.astype(np.uint8)

    def __getitem__(self, idx):
        A_path = self.A_paths[idx % self.A_size]
        A_img = Image.open(A_path).convert('RGB')
        A_img = self.resize_and_crop(A_img) # Resize and crop image
        A_img = Image.fromarray(self.set_im_mean(np.array(A_img))) # Normalize image mean
        A = self.transform_A(A_img) # Do the other transforms we have

        if self.phase == 'train':
            B_path = self.B_paths[random.randint(0, self.B_size-1)] # choose random image
            B_img = Image.open(B_path).convert('RGB')
            B_img = self.resize_and_crop(B_img) # Resize and crop image
            B_img = Image.fromarray(self.set_im_mean(np.array(B_img))) # Normalize image mean
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
        transform_list.append(transforms.Grayscale(3)) # 3 channel grayscale image for identity L1 loss
    transform_list.append(transforms.RandomHorizontalFlip()) # Random flip
    transform_list.append(transforms.ToTensor()) # Convert to tensor
    transform_list.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))) # Normalize values between -1 and 1
    return transforms.Compose(transform_list)

