from flask import Flask, request, jsonify
from flask_cors import CORS
import cycleganime
import time
import utils
import redis
from rq import Queue
import numpy as np
import cv2
from b2sdk.v1 import InMemoryAccountInfo, B2Api
import config
import os
import io
from PIL import Image


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # Max 10MB image
cors = CORS(app, resources={r"/api/*": {"origins": "https://cycleganime.drew.hu"}})

# Redis task queue
r = redis.Redis()
q = Queue(connection=r)

# Load model
model = cycleganime.CycleGANime(n_blocks=15, ngf=128)
model.load_weights("models/pink.pth")
transform = utils.get_transforms() # Image transforms

# Keep track of results (they expire after 24 hours)
# result_ids, result_q = utils.load_result_ids()

def run_inference(im,  result_code):
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

@app.route('/api/paint', methods=['POST'])
def _paint():
    if 'image' not in request.files:
        return "No file", 206

    if not utils.verify_recaptcha(request.form['recaptcha']):
        return 'Invalid recaptcha', 206

    file = request.files['image']

    # Read image from request data
    try:
        im = Image.open(io.BytesIO(bytearray(file.read()))).convert('RGB')
        im = im.resize((256,256), Image.BICUBIC)
        im = Image.fromarray(utils.set_im_mean(np.array(im), mean=185))
        im = transform(im)
    except:
        return "Couldn't read image", 206


    # Create unique code for this result
    result_id = utils.create_code()

    # Add result id to result ids
    # result_ids.add(result_id)

    # Submit job to task queue
    q_start = time.time()
    q.enqueue(run_inference, args=(im, result_id))
    q_duration = time.time()-q_start
    with open("log.txt", "a") as f:
        f.write(f"Queue: {q_duration:.2f}s")

    return jsonify(result_id)

# @app.route('/api/check/<result_id>', methods=['GET'])
# def _check(result_id):
#     """Returns whether or not a result id exists"""
#     # Clear expired images from result queue
#     while True:
#         # Check if timestamp is older than 24 hours
#         if len(result_q) == 0:
#             break
#         result_id_, upload_timestamp = result_q[0]
#         if upload_timestamp < (time.time() - 24*3600)*1000:
#             result_ids.remove(result_id_)
#             result_q.popleft()
#         else:
#             break
#     return jsonify(result_id in result_ids)
