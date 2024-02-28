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
hide_rest_value = False # If true, only the parts recognised by LIME are displayed and the rest is hidden
show_pros_cons = True
show_pros_cons_value = 10 # 10 if they want pros and cons, 5 if not
def explain_image(img, predicted_class):

    explanation = explainer.explain_instance(img[0], deep_model.predict, top_labels=5,
                                             hide_color=0, num_samples=1000)
    if show_pros_cons == True:
        show_pros_cons_value = 10
    else:
        show_pros_cons_value = 5


    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=False,
                                                num_features=show_pros_cons_value,
                                                hide_rest=hide_rest_value,
                                                min_weight=0.1)

    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    explained_image = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title("Predicted Class: " + predicted_class)
    plt.show()

    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

    # plt.imshow(heatmap, cmap = 'RdBu', vmin = -heatmap.max(), vmax = heatmap.max())
    # plt.colorbar()
    # plt.show()

    return heatmap, explained_image
# test_image = os.path.join(test_data_directory, "test2.jpg")
# explain_image(test_image)