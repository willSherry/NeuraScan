import os
import numpy as np
import pandas as pd
import PIL
import tqdm
from PIL import UnidentifiedImageError
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import tensorflow
from medpy.filter.smoothing import anisotropic_diffusion
from tqdm import tqdm

from train_model import create_and_train_model
def preprocess_image(img):
    # Normalizing intensity
    normalized_image = intensity_normalization(img)
    # Removing noise
    noise_removed_image = noise_removal(normalized_image)

    return noise_removed_image

def intensity_normalization(img):
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_image = (img - min_val) / (max_val - min_val)
    return normalized_image

def noise_removal(img):
    gaussian_img = cv2.GaussianBlur(img, (5,5), 0)
    median_img = cv2.medianBlur(gaussian_img, 5)
    #diffused_img = anisotropic_diffusion(median_img)
    return median_img

# Getting directories
project_directory = os.getcwd()
data_directory = os.path.join(project_directory, 'Data')

# Dividing pixel values by 255 and creating validation split
print('Dividing pixel values by 255...')
datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                  validation_split=0.2)

# Loading images and splitting 80/20 for training and validation
print('Loading images...')
train_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

def preprocess_generator(generator):
    preprocessed_images = []
    labels = []
    iterations = 0
    numberOfImages = len(generator)
    for images, batch_labels in tqdm(generator, desc="Preprocessing images", total=numberOfImages):
        preprocessed_batch = []
        for img in images:
            iterations += 1
            preprocessed_batch.append(preprocess_image(img))
            if iterations >= numberOfImages:
                break
        preprocessed_images.append(np.array(preprocessed_batch))
        labels.extend(batch_labels)
        if iterations >= numberOfImages:
            break
    return np.concatenate(preprocessed_images), np.array(labels)

print("Preprocessing training images...")
preprocessed_training_images = preprocess_generator(train_generator)

print("Preprocessing validation images...")
preprocessed_validation_images = preprocess_generator(validation_generator)

def generate_data(generator):
    while True:
        images, labels = next(generator)
        yield {'conv2d_input': images}, labels

# Train the model using custom data generators
train_data_generator = generate_data(train_generator)
validation_data_generator = generate_data(validation_generator)

train_gen_length = len(train_generator)
validation_gen_length = len(validation_generator)

create_and_train_model(train_data_generator, validation_data_generator,
                       train_gen_length, validation_gen_length)
