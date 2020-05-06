"""
Downloads images (concurrently!)

Optional:
    Duplicate image checking (hash)
    Image resizing 
    Color vs. greyscale detection

"""

from concurrent.futures import ThreadPoolExecutor
import requests
import os
import os.path
import untangle
import hashlib
import numpy as np
import cv2
import time
from skimage import io

# names = [
#     'hatsune_miku', # 3510 (vocaloid blue hair)
#     'kaname_madoka', #261
#     'kaname_madoka', #261

#     'kirisame_marisa', #835
#     'remilia_scarlet', #1102
#     'hakurei_reimu', #1297
#     'patchouli_knowledge', #595
# ]
SAVE_PATH = "data/pink_hair/trainB"

# TAGS = ['lineart', 'solo', 'monochrome']
TAGS = ['pink_hair', 'solo', '1girl', 'white_background', 'simple_background', 'long_hair']
#TAGS = ["hatsune_miku", "blue_hair", "1girl", "solo", "white_background"]
# IM_SIZE = 512  # Downloaded image size
NUM_WORKERS=6 # Number of workers for threads

urls = set()
im_hashes = set()

base_url = "http://safebooru.org/index.php?page=dapi&s=post&q=index&tags={}&pid=".format('%20'.join(TAGS))

def get_urls(pid):
    """Stores image URLs for the page ID in urls"""
    global urls
    r = requests.get(base_url+str(pid))
    xml = untangle.parse(r.text)

    # If there are no posts
    if len(xml.posts.get_elements()) == 0:
        return

    # Add urls to list
    for post in xml.posts.post:
        url = post['file_url']
        if ('png' in url) or ('jpg' in url):
            urls.add(url)

def download_image(url):
    """Downloads and saves an image"""
    # global images_downloaded

    # Download image
    im = io.imread(url)

    # Create filename
    file_name = os.path.splitext(url[url.rindex('/')+1:])[0]+".jpg"

    # Check if we already downloaded this image
    # im_hash = sha256sum(im.data)
    # if im_hash in im_hashes:
    #     return
    # else:
    #     im_hashes.add(im_hash)
    
    # If image is RGBA, make the transparent channel all white,
    # or else the image will appear as black
    if im.shape[-1] == 4:
        im[im[...,-1]==0] = [255,255,255,0]
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB) # convert to 3 channel RGB

    # Convert BGR to RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Resize image
    # h, w = im.shape[:2]
    # if h > w:
    #     rat = IM_SIZE / w
    #     new_h, new_w = h * rat, IM_SIZE
    # else:
    #     rat = IM_SIZE / h
    #     new_h, new_w = IM_SIZE, w * rat
    # res = cv2.resize(im, (int(new_w), int(new_h)), interpolation=cv2.INTER_CUBIC)

    # Modify save path based on whether image is color or not
    # SAVE_PATH = os.path.join(SAVE_PATH, 'color' if is_color(im) else 'bw')

    cv2.imwrite(os.path.join(SAVE_PATH, file_name), im)
    # images_downloaded += 1

def sha256sum(data):
    """Calculate hash of image"""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def is_color(im, thumb_size=128, MSE_cutoff=22):
    """Detects if an image is color or not
    Adapted from https://stackoverflow.com/questions/20068945/
    """
    im = cv2.resize(im, (thumb_size, thumb_size))
    channels = cv2.split(im)[:3]
    if len(channels) < 3: # 1d images are black and white
        return False
    im_mean = np.mean(im) # Mean across all pixels

    # Adjust color bias
    bias = np.array([np.mean(c) - im_mean for c in channels])
    channel_mean = np.mean(im, axis=2) # Mean across each channel
    SSE = np.sum(np.square(im - np.expand_dims(channel_mean, axis=2)-bias))
    MSE = SSE / thumb_size**2
    return MSE > MSE_cutoff
    
if __name__ == "__main__":
    print("Downloading images with tags", TAGS)

    print("Fetching urls from base url", base_url)
    
    # See how many pages we'll have to download
    r = requests.get(base_url+str(0))
    xml = untangle.parse(r.text)
    page_count = int(xml.posts['count'])//100

    # Fetch urls
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(get_urls, list(range(page_count)))
    time_elapsed = time.time()-start_time
    print("Fetched %d urls in %0.1f seconds" % (len(urls), time_elapsed))

    # Get list of existing images so we don't download what we already have
    existing_ims = set([os.path.splitext(x)[0] for x in os.listdir(SAVE_PATH)])
    urls = [x for x in urls if os.path.splitext(os.path.basename(x))[0] not in existing_ims]
    print(f"Found {len(urls)} new images")

    # Download images
    print("Downloading %d images..." % len(urls))
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(download_image, list(urls))
    print("Downloaded images in %0.1f seconds" % (time.time()-start_time))

