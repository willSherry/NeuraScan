import os
import numpy as np
import pandas as pd
import PIL
from PIL import UnidentifiedImageError
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from medpy.filter.smoothing import anisotropic_diffusion
def load_and_preprocess_data(data_directory):
    filepaths = list(Path(data_directory).rglob('*.jpg'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    image_df = pd.concat([filepaths, labels], axis=1)

    # Divide pixel values by 255
    print('Dividing pixel values by 255...')
    images = []
    for filepath in filepaths:
        img = PIL.Image.open(filepath)
        img = img.resize((224, 224))
        img = np.asarray(img) / 255.0
        images.append(img)

    image_df['Image'] = images

    # Intensity Normalization
    print('Normalizing intensity...')
    image_df['Image'] = image_df['Image'].apply(intensity_normalization)

    # Noise Removal
    print('Removing noise...')
    image_df['Image'] = image_df['Image'].apply(noise_removal)
    return image_df

def intensity_normalization(img):
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_image = (img - min_val) / (max_val - min_val)
    return normalized_image

def noise_removal(img):
    gaussian_img = cv2.GaussianBlur(img, (5,5), 0)
    median_img = cv2.medianBlur(gaussian_img, 5)
    diffused_img = anisotropic_diffusion(median_img)
    return diffused_img

data_directory = os.path.join(os.getcwd(), 'Data')
image_df = load_and_preprocess_data(data_directory)