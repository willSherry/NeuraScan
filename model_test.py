import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocess_data import preprocess_image

project_directory = os.getcwd()
data_directory = os.path.join(project_directory, 'Data')

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def predict_single_image(model, image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    preprocessed_img = preprocess_image(img)

    preprocessed_img = cv2.resize(preprocessed_img, target_size)

    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

    preprocessed_img = np.repeat(preprocessed_img, 1, axis=-1)  # Repeat along the channel dimension to have 3 channels

    predictions = model.predict(preprocessed_img)

    class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']  # Define your class labels here
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class_label, confidence





if __name__ == "__main__":
    model_path = os.path.join(project_directory,"classificationModel.keras")

    image_path = os.path.join(data_directory,"ModerateDemented\\0b1b32a8-553c-4eaa-a356-1226d90dc031.jpg")

    model = load_trained_model(model_path)

    predicted_class, confidence = predict_single_image(model, image_path)

    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)

# Current issue, it works, but the predictions are completely wrong
# Make the images bigger? 256 x 256? Worst comes to worst remove all preprocessing
# and just train it from unprocessed data