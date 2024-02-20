import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras import regularizers
import math

def create_and_train_model(train_data_generator, validation_data_generator, train_len, val_len):
    model = tf.keras.models.Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(128,128, 3), kernel_regularizer=regularizers.l1_l2(l1=0.000001, l2=0.000001)),
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

    history = model.fit(
        train_data_generator,
        steps_per_epoch=train_len,
        validation_data=validation_data_generator,
        validation_steps=val_len,
        epochs=10
    )

    model.save("classificationModel.keras")
