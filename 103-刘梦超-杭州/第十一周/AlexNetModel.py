#!/usr/bin/env python 
# coding:utf-8

"""
1、一张原始图片被resize到(224,224,3);
2、使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层， 输出的shape为(55,55,96);
3、使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
4、使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层， 输出的shape为(27,27,256);
5、使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256);
6、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层， 输出的shape为(13,13,384);
7、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层， 输出的shape为(13,13,384);
8、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层， 输出的shape为(13,13,256);
9、使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256);
10、两个全连接层，最后输出为1000类
"""
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dense, Flatten, Dropout


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    # 构建模型
    model = Sequential()
    # 卷积
    model.add(
        Conv2D(
            strides=(4, 4),
            kernel_size=(11, 11),
            filters=48,
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    # 批标准化
    model.add(BatchNormalization())
    # 添加激活函数
    # model.add(Activation('relu'))
    # 最大值池化
    model.add(
        MaxPooling2D(
            strides=(2, 2),
            pool_size=(3, 3),
            padding='valid'
        )
    )
    # 卷积
    model.add(
        Conv2D(
            strides=(1, 1),
            kernel_size=(5, 5),
            filters=128,
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    # 最大值池化
    model.add(
        MaxPooling2D(
            strides=(2, 2),
            pool_size=(3, 3),
            padding='valid'
        )
    )
    # 卷积
    model.add(
        Conv2D(
            strides=(1, 1),
            kernel_size=(3, 3),
            filters=192,
            padding='same',
            activation='relu'
        )
    )
    # 卷积
    model.add(
        Conv2D(
            strides=(1, 1),
            kernel_size=(3, 3),
            filters=192,
            padding='same',
            activation='relu'
        )
    )
    # 卷积
    model.add(
        Conv2D(
            strides=(1, 1),
            kernel_size=(3, 3),
            filters=128,
            padding='same',
            activation='relu'
        )
    )
    # 最大值池化
    model.add(
        MaxPooling2D(
            strides=(2, 2),
            pool_size=(3, 3),
            padding='valid'
        )
    )
    # 数据打平
    model.add(Flatten())
    # 全连接层
    model.add(
        Dense(1024, activation='relu')
    )
    # 防止过拟合,随机去除一些节点
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    # 最终输出两类
    model.add(Dense(output_shape, activation='softmax'))
    return model
