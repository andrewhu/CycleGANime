import torchvision.transforms as transforms
from PIL import Image
import random
import requests
import json
import os
import os.path
import cv2
import numpy as np
import config
from b2sdk.v1 import InMemoryAccountInfo, B2Api
import io
import time
from collections import deque

def load_result_ids():
    """Returns set of file ids and a queue ordering them"""
    result_ids = set()
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", config.B2_KEY, config.B2_SECRET)
    bucket = b2_api.get_bucket_by_name(config.B2_BUCKET)
    files = []
    for file_info, _ in bucket.ls(folder_to_list="cycleganime/results",show_versions=False, recursive=True):
        if file_info.content_type != "image/jpeg":
            continue
        result_id = os.path.basename(file_info.file_name)[:5]
        upload_timestamp = file_info.upload_timestamp
        result_ids.add(result_id)
        files.append((result_id, upload_timestamp))
    files = sorted(files, key=lambda x: x[1]) # Sort by upload timestamp
    result_q = deque(files)
    return result_ids, result_q


def verify_recaptcha(captcha):
    """Verifies recaptcha response"""
    response = requests.post('https://www.google.com/recaptcha/api/siteverify',
                             data={'secret': config.RC_SECRET,
                                   'response': captcha})
    return json.loads(response.text)['success']

def get_transforms():
    """Image transforms"""
    transform_list = [
        transforms.Grayscale(3),  # 3 channel grayscale image for identity L1 loss
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]
    return transforms.Compose(transform_list)

def set_im_mean(im, mean=200, threshold=250, max_iters=50):
    """
    Iteratively brightens or darkens an image until it reaches a 
    chosen mean intensity value. This is so that the average pixel
    intensity of the inputs are the same, which improves quality.

    Consider this as a type of normalization
    """
    im = im.astype(np.float32)
    
    # Mean of non-white pixels
    initial_mean = im[im<threshold].mean()

    # Iteratively brighten or darken non-white pixels
    if initial_mean < mean: # image is darker
        for _ in range(max_iters):
            im_mean = im[im<threshold].mean()
            if im_mean > mean: # If our image is bright enough
                break
            x = abs(mean - im_mean)/255 + 0.005
            im[im<threshold] = im[im<threshold]*(1-x)+255*x # Brighten image
    else: # image is brighter
        for _ in range(max_iters):
            im_mean = im[im<threshold].mean()
            if im_mean < mean: # If our iamge is dark enough
                break
            im[im<threshold] = im[im<threshold] * 0.995 - 255 * .005 # Darken image

    return im.astype(np.uint8)

def create_code():
    """Generates unique 5-letter code"""
    alphabet = 'BCDFGHJLMNPQRSTVWXZ' # Letters excluding vowels and K
    return ''.join(random.choices(alphabet, k=5))

def read_and_transform_image(file, transform):
    """Reads in image data from request"""
    try:
        im = Image.open(io.BytesIO(bytearray(file.read()))).convert('RGB')
        im = im.resize((256,256), Image.BICUBIC)
        im = Image.fromarray(set_im_mean(np.array(im), mean=185))
        im = transform(im)
    except:
        return False
    return im


def run_inference(im,  model, result_code):
    # Perform inference
    inference_start = time.time()
    result = model.run_inference(im)[0]
    inference_duration = time.time()-inference_start
    result = cv2.cvtColor((result*255).astype(np.float32), cv2.COLOR_BGR2RGB)

    # Save image
    pink_path = f"results/{result_code}.jpg"
    cv2.imwrite(pink_path, result)

    # Upload to b2
    upload_start = time.time()
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", config.B2_KEY, config.B2_SECRET)
    bucket = b2_api.get_bucket_by_name(config.B2_BUCKET)
    bucket.upload_local_file(local_file=pink_path, file_name=f"cycleganime/results/{os.path.basename(pink_path)}")
    upload_duration = time.time()-upload_start

    # Remove local file
    os.remove(pink_path)

    # Log output
    with open("log.txt", "a") as f:
        f.write(f"{pink_path}\t{inference_duration:.2f}\t{upload_duration:.2f}\n")