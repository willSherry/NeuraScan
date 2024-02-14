import os
import sys

import numpy as np
import cv2
from tensorflow.keras.models import load_model

import lime_integration
from preprocess_data import preprocess_image

project_directory = os.getcwd()
test_data_directory = os.path.join(project_directory, 'TestData')
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def preprocess_single_image(image_path):
    img = cv2.imread(image_path)
    target_size = (128, 128)

    preprocessed_image = cv2.resize(img, target_size)
    preprocessed_image = np.array(preprocessed_image) / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    preprocessed_image = np.repeat(preprocessed_image, 1, axis=-1)

    return preprocessed_image

def predict_single_image(model, image_path):

    preprocessed_image = preprocess_single_image(image_path)
    predictions = model.predict(preprocessed_image)

    class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']  # Define your class labels here
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class_label, confidence

if __name__ == "__main__":
    model_path = os.path.join(project_directory,"classificationModel.keras")

    image_path = os.path.join(test_data_directory,"test2.jpg")


    model = load_trained_model(model_path)

    try:
        predicted_class, confidence = predict_single_image(model, image_path)
    except:
        print("Try again with a valid image")
        sys.exit()

    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)

    lime_integration.explain_image(image_path, predicted_class)