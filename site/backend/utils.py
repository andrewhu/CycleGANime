import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import random
import requests
import json
import config

def verify_recaptcha(captcha):
    """Verifies recaptcha response"""
    response = requests.post('https://www.google.com/recaptcha/api/siteverify',
                             data={'secret': config.RC_SECRET,
                                   'response': captcha})
    return json.loads(response.text)['success']

def get_transforms():
    """Image transforms"""
    transform_list = []
    transform_list.append(transforms.Grayscale(3)) # 3 channel grayscale image for identity L1 loss
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
    initial_mean = im[im<250].mean()

    # Iteratively brighten or darken non-white pixels
    if initial_mean < mean: # image is darker
        for _ in range(max_iters):
            im_mean = im[im<250].mean()
            if im_mean > mean: # If our image is bright enough
                break
            x = abs(mean - im_mean)/255 + 0.005
            im[im<250] = im[im<250]*(1-x)+255*x # Brighten image
    else: # image is brighter
        for _ in range(max_iters):
            im_mean = im[im<250].mean()
            if im_mean < mean: # If our iamge is dark enough
                break
            im[im<250] = im[im<250] * 0.995 - 255 * .005 # Darken image

    return im.astype(np.uint8)

def create_code():
    """Generates unique 5-letter code"""
    alphabet = 'BCDFGHJLMNPQRSTVWXZ' # Letters excluding vowels and K
    return ''.join(random.choices(alphabet, k=5))





