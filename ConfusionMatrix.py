import os
import numpy as np
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Getting directories
project_directory = os.getcwd()
parent_directory = os.path.dirname(project_directory)
data_directory = os.path.join(parent_directory, 'TestData\\Alzheimer_MRI_4_classes_dataset')

model_path = os.path.join(project_directory, "classificationModel.keras")

testing_datagen = ImageDataGenerator(rescale=1. / 255)
testing_generator = testing_datagen.flow_from_directory(
    data_directory,
    target_size=(128, 128),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)

model = load_model(model_path)

batch_size = 100
target_names = ['0', 'R']
Y_pred = model.predict(testing_generator)
y_pred = np.argmax(Y_pred, axis=1)
print("Confusion Matrix")
cm = confusion_matrix(testing_generator.classes, y_pred)
print(cm)
