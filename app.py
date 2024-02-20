import app
import cv2
from flask import Flask, request, jsonify, render_template

from tensorflow import keras
from keras.models import load_model

from model_test import preprocess_single_image

app = Flask(__name__)

image_model = load_model('classificationModel.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Passing the image to the model for prediction
    print('We got here at least')
    scan = preprocess_single_image(file)
    prediction = image_model.predict(scan)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
