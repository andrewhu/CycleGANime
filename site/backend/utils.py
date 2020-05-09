import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def verify_recaptcha(recaptcha):
    pass

def get_transforms():
    """Image transforms"""
    transform_list = []
    transform_list.append(transforms.Grayscale(3)) # 3 channel grayscale image for identity L1 loss
    # transform_list.append(transforms.RandomHorizontalFlip()) # Random flip
    transform_list.append(transforms.ToTensor()) # Convert to tensor
    transform_list.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))) # Normalize 
    return transforms.Compose(transform_list)

def set_im_mean(im, mean=200, max_iters=50):
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
            im_mean = im[np.where(im<254)].mean()
            if im_mean > mean: # If our image is bright enough
                break
            x = abs(mean - im_mean)/255 + 0.005
            im[im<254] = im[im<254]*(1-x)+255*x # Brighten image
    else: # image is brighter
        for _ in range(max_iters):
            im_mean = im[np.where(im<254)].mean()
            if im_mean < mean: # If our iamge is dark enough
                break
            im[im<254] = im[im<254] * 0.995 - 255 * .005 # Darken image

    return im.astype(np.uint8)


# def transform_image(im):
#     im = set_im_mean(im)

#     im = Image.fromarray(im)
#     transform = get_transforms()
#     im = transform(im)
#     return im
    




