import cycleganime
import numpy as np 
import cv2
import torch
import time
import utils
import glob
from PIL import Image
import os.path

print("Loading models...")
models = []
for model_path in glob.glob("models/*.pth"):
    print(f"Loading '{model_path}'...", end=' ')
    model = cycleganime.CycleGANime(n_blocks=15,ngf=128)
    model.load_weights(model_path)
    models.append(model)
    print("done")

print("Loading transforms...", end=' ')
transform = utils.get_transforms()
print("done")

print("Starting inference")
start_time = time.time()
im_paths = glob.glob("test_images/*.jpg")
for im_path in im_paths:
    print(im_path)
    im = Image.open(im_path).convert('RGB')
    im = im.resize((256,256), Image.BICUBIC)
    im = Image.fromarray(utils.set_im_mean(np.array(im), mean=185))
    im = transform(im)
    basename = os.path.basename(im_path)
    for idx, model in enumerate(models):
        print(f"model {idx} starting...")
        start_time = time.time()
        result = model.run_inference(im)[0]
        time_elapsed = time.time()-start_time
        result = cv2.cvtColor((result*255).astype(np.float32), cv2.COLOR_BGR2RGB)
        print(f"Done in {time_elapsed}")
        cv2.imwrite(os.path.join("test_out", f"{idx}_{basename}"), result)



