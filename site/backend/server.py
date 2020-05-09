from flask import Flask, request
from flask import jsonify, make_response
from flask_cors import CORS
import cycleganime

import numpy as np 
import cv2
import torch
import time
import utils
from PIL import Image


app = Flask(__name__)
# app.secret_key = config.FLASK_SECRET_KEY
# app.config['MAX_CONTENT_']
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

model = cycleganime.CycleGANime()
model.load_weights("netG_A_latest.pth")

transform = utils.get_transforms()

@app.route('/api/paint', methods=['POST'])
def _paint():
    start = time.time()
    if 'file' not in request.files:
        return "No file"
    file = request.files['file']

    # Read in image data
    im = np.asarray(bytearray(file.read()), dtype=np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    h, w = im.shape[:2]

    # Resize image
    im = cv2.resize(im, (512,512))
    
    # normalize image mean
    im = utils.set_im_mean(im)

    # Torchvision transforms
    im = Image.fromarray(im)
    im = transform(im)
    
    # Run inference
    result = model.run_inference(im)[0]*255
    result = cv2.resize(result, (w, h))
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Save image
    cv2.imwrite("test.jpg", result)
    time_elapsed = time.time()-start
    with open("log.txt", "a") as f:
        f.write(str(time_elapsed) + '\n')

    return 'ok'

