import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from model_test import predict_single_image
from lime_integration import explain_image
from PIL import Image

app = Flask(__name__)
model = load_model("classificationModel.keras")
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    image_file= request.files['imagefile']

    img = Image.open(image_file)
    img = img.resize((128,128))
    img = np.array(img)
    img = img / 255.0

    if img.shape[-1] == 128:
        img = np.expand_dims(img, axis=-1)
    if img.shape[0] == 128:
        img = np.expand_dims(img, axis=0)
    if img.shape[-1] != 3:
        img = np.repeat(img, 3, axis=3)

    prediction, confidence = predict_single_image(model, img)

    return render_template('index.html', prediction=prediction, confidence=round(confidence*100, 4))

if __name__ == '__main__':
    app.run(port=3000, debug=True)