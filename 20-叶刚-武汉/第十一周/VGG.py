"""
基于tensorflow.keras的VGG16分类网络实现
tensorflow: 2.3.0
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# 卷积核个数减半，全连接层神经元个数由4096减为512
def VGG16(input_shape: tuple = (224, 224, 3), num_classes: int = 1000):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), strides=(1, 1), padding='same', input_shape=input_shape, activation='relu'),
        Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),

        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),

        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),

        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),

        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),

        Flatten(),
        Dense(512, activation='relu'),     # 论文中是4096，此处改小为512
        Dense(512, activation='relu'),     # 论文中是4096，此处改小为512
        Dense(num_classes)
    ])
    return model

