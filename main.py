import cv2
import tensorflow as tf
import os
import imghdr
from matplotlib import pyplot as plt

# Setting GPU memory comsumption growth and avoiding out of memory error
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

project_directory = os.getcwd()

# Directory for the dataset
data_dir = os.path.join(project_directory, 'Data')

image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# ModerateDementedData = os.path.join(data_dir, 'Moderate_Demented')

# Removing any dodgy images from the datasets
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(128, 128))
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
# Images represented as numpy arrays
batch[0].shape

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

3326
# STILL HAVE TO COMMIT TO GITHUB