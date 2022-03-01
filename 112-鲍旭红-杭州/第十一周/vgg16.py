#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/24 20:35
# @Author  : bxh
# @Email   : 573878341@qq.com
# @File    : vgg16.py



from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

def vgg16(input_shape = (224,224,3),output_shape = 2):
    model = Sequential()
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            strides=1,
            padding='same',
            input_shape=input_shape,
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(
        Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(
        Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Flatten())
    #全链接层
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.80))
    model.add(Dense(output_shape,activation='softmax'))
    return model