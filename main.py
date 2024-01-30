import os
import numpy
import numpy as np
import pandas as pd
import imghdr
import cv2
import tensorflow as tf
from sklearn.utils import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, losses
import seaborn as sns
import matplotlib.pyplot as plt

import PIL
from PIL import UnidentifiedImageError
from pathlib import Path
import os.path

import itertools
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras import regularizers

# Getting data -- MAY HAVE TO CHANGE THIS
project_directory = os.getcwd()
data_directory = os.path.join(project_directory, 'Data')
image_directory = Path(data_directory)

filepaths = list(image_directory.glob(r'**/*.JPG')) + list(image_directory.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

image_df = pd.concat([filepaths, labels], axis=1)
path = image_directory.glob("*.jpg")
for img_p in path:
    try:
        img_p = PIL.Image.open(img_p)
    except PIL.UnidentifiedImageError:
        print(img_p)

#Displaying 16 x 16 image of the data with labels
random_index = np.random.randint(0, len(image_df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
    ax.set_title(image_df.Label[random_index[i]])
plt.tight_layout()
plt.show()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2
                                   )
# Setting validation split

# Setting training data
train_images = train_datagen.flow_from_directory(
    data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Setting validation data
validation_images = train_datagen.flow_from_directory(
    data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
model = tf.keras.models.Sequential([
    Conv2D(16, (3,3), activation = 'relu', input_shape = (224,224, 3), kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
    Dropout(0.2),
    Dense(4, activation = 'softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])
history = model.fit(train_images,
    steps_per_epoch=len(train_images),
    validation_data=validation_images,
    validation_steps=len(validation_images),
    epochs=10)

model.save("classificationModel.keras")