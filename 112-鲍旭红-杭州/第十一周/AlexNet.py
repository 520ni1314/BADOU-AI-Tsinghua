#!/usr/bin/env python
# encoding: utf-8
'''
@author: 醉雨成风
@contact: 573878341@qq.com
@software: python
@file: AlexNet.py
@time: 2022/2/22 22:11
@desc:
'''


from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam


def AlexNet(input_shape = (224,224,3),output_shape = 2):
    model = Sequential()
    model.add(
    Conv2D(
        filters=48,
        kernel_size=(11,11),
        strides=(4,4),
        padding='valid',
        input_shape=input_shape,
        activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(Conv2D(
        filters=128,
        kernel_size=(5,5),
        strides=(1,1),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(Conv2D(
        filters=192,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    ))
    model.add(Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    ))
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    ))
    # model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    #展平
    model.add(Flatten())
    #全连接层
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.85))
    model.add(Dense(output_shape,activation='softmax'))
    return model
