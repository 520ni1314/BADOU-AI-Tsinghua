"""
基于tensorflow.keras的AlexNet分类网络实现
tensorflow: 2.3.0
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


def AlexNet(input_shape: tuple = (224, 224, 3), num_classes: int = 1000):
    # 为了加快收敛，将每个卷积层的filter数量减半，全连接层减为1024、512
    model = tf.keras.Sequential([
        Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding='valid',
               input_shape=input_shape, activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        Conv2D(128, (5, 5), (1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        Conv2D(192, (3, 3), (1, 1), padding='same', activation='relu'),
        Conv2D(192, (3, 3), (1, 1), padding='same', activation='relu'),
        Conv2D(128, (3, 3), (1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(num_classes)
    ])
    return model

