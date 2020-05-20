from flask import Flask, request
from flask import jsonify, make_response, send_file
from flask_cors import CORS
import cycleganime
import numpy as np 
import cv2
import torch
import time
import utils
import glob
from PIL import Image
import io
import threading


app = Flask(__name__)
# app.secret_key = config.FLASK_SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # Max 10MB image
cors = CORS(app, resources={r"/api/*": {"origins": "https://cycleganime.drew.hu"}})

models = []
for model_path in glob.glob("models/*.pth"):
    model = cycleganime.CycleGANime(n_blocks=15, ngf=128)
    model.load_weights(model_path)
    models.append(model)

transform = utils.get_transforms()

@app.route('/api/paint', methods=['POST'])
def _paint():
    if 'image' not in request.files:
        return "No file"

    # if not utils.verify_recaptcha(request.form['recaptcha']):
    #     return 'Invalid recaptcha'

    file = request.files['image']

    try:
        im = Image.open(io.BytesIO(bytearray(file.read()))).convert('RGB')
        im = im.resize((256,256), Image.BICUBIC)
        im = Image.fromarray(utils.set_im_mean(np.array(im), mean=185))
        im = transform(im)
    except:
        return "Couldn't read image", 206

    # Create unique code for this result
    result_code = utils.create_code()

    def run_models():
        # Run inference
        for idx, model in enumerate(models):
            start_time = time.time()
            result = model.run_inference(im)[0]
            time_elapsed = time.time()-start_time
            result = cv2.cvtColor((result*255).astype(np.float32), cv2.COLOR_BGR2RGB)

            # Save image
            cv2.imwrite(f"results/{result_code}_{idx+1}.jpg", result)

            with open("log.txt", "a") as f:
                f.write(f"{result_code}_{idx+1} {time_elapsed}\n")


    threading.Thread(target=run_models).start()

    return result_code #send_file(fn, attachment_filename="test.jpg")

