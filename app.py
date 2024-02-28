import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from model_test import preprocess_single_image
from model_test import predict_single_image
from lime_integration import explain_image
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = load_model("classificationModel.keras")

img = st.file_uploader(label="Upload or drag brain scan here!", type=['png', 'jpg', 'jpeg'])
#img = 'D:\\Final Year Project\\Project\\NeuraScan\\TestData\\moderateDemented1.jpg'

if img is not None:
    img = Image.open(img)
    st.image(img, use_column_width=True)

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

    prediction, confidence = predict_single_image(model, img)

    st.write(f"Prediction is: {prediction} \nConfidence: {confidence*100}%")

    heatmap, explained_image = explain_image(img, prediction)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    heatmap_plot = axes[0].imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    axes[0].set_title('Heatmap')

    regular_plot = axes[1].imshow(explained_image)
    axes[1].set_title('Explained Image')
    fig.colorbar(heatmap_plot, ax=axes[0])

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Prediction: {prediction}\nConfidence: {round(confidence*100, 2)}%")

    st.pyplot(fig)