import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocess_data import image_df


def create_and_train_model(train_images, validation_images):
    model = tf.keras.models.Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(224,224, 3), kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
        Dropout(0.2),
        Dense(4, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit(train_images,
                        steps_per_epoch=len(train_images),
                        validation_data=validation_images,
                        validation_steps=len(validation_images),
                        epochs=10)

    model.save("classificationModel.keras")


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_images = train_datagen.flow_from_directory(image_df['Normalized_Denoised_Image'], target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
validation_images = train_datagen.flow_from_directory(image_df['Normalized_Denoised_Image'], target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')
create_and_train_model(train_images, validation_images)
