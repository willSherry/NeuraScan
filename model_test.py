import os
import sys

import numpy as np
import cv2
import tensorflow
from tensorflow import keras
from keras.models import load_model
from PIL import Image

import lime_integration
from preprocess_data import preprocess_image

project_directory = os.getcwd()
test_data_directory = os.path.join(project_directory, 'TestData')
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def predict_single_image(model, image_path):

    img = Image.open(image_path)
    img = img.resize((128, 128))
    img = np.asarray(img)
    img = img / 255.0

    # CHECK IF IT ALREADY HAS ALL THE DIMS BEFORE DOING THIS
    if img.shape[-1] == 128:
        img = np.expand_dims(img, axis=-1)
    if img.shape[0] == 128:
        img = np.expand_dims(img, axis=0)
    if img.shape[-1] != 3:
        img = np.repeat(img, 3, axis=3)

    predictions = model.predict(img)

    class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'] 
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class_label, confidence, img

if __name__ == "__main__":
    model_path = os.path.join(project_directory,"classificationModel.keras")

    image_path = os.path.join(test_data_directory,"moderateDemented3.jpg")


    model = load_trained_model(model_path)

    try:
        predicted_class, confidence = predict_single_image(model, image_path)
    except:
        print("Try again with a valid image")
        sys.exit()

    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)

    lime_integration.explain_image(image_path, predicted_class)