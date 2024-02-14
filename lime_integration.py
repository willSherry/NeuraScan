import cv2
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries

import model_test

project_directory = os.getcwd()
test_data_directory = os.path.join(project_directory, "TestData")
model_path = os.path.join(project_directory, "classificationModel.keras")

deep_model = load_model(model_path)
explainer = lime_image.LimeImageExplainer()

# LIME result options
hide_rest_value = False
show_pros_cons = True
show_pros_cons_value = 10 # 10 if they want pros and cons, 5 if not
def explain_image(img, predicted_class):
    img = Image.open(img)
    target_size = (128, 128)

    preprocessed_image = img.convert("RGB")
    preprocessed_image = np.array(preprocessed_image) / 255.0
    preprocessed_image = cv2.resize(preprocessed_image, target_size)
    if len(preprocessed_image.shape) == 3:
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    preprocessed_image = np.repeat(preprocessed_image, 1, axis=-1)

    explanation = explainer.explain_instance(preprocessed_image[0], deep_model.predict, top_labels=5,
                                             hide_color=0, num_samples=1000)
    if show_pros_cons == True:
        show_pros_cons_value = 10
    else:
        show_pros_cons_value = 5


    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=False,
                                                num_features=show_pros_cons_value,
                                                hide_rest=hide_rest_value,
                                                min_weight=0.15)

    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title("Predicted Class: " + predicted_class)
    plt.show()

# test_image = os.path.join(test_data_directory, "test2.jpg")
# explain_image(test_image)