# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/5/8

from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

# 定义网络结构函数，参数包括输入数据的shape和输出的shape
'''
设输入图片大小为w1*w1*C1，卷积核大小为F*F，卷积核个数为N（卷积核个数跟卷积核通道数无关，卷积核个数决定输出的通道数，
而卷积核通道数由输入数据的通道数决定），步长为S，padding的数量（单边）为P，设输出大小为w2*w2*c2,那么，一定有输出数据的通道数C2=N。
且当padding模式为‘valid’时，有w2=(w1-F)/S+1.这个公式对池化层同样成立，但池化不改变通道数。
当padding模式为'same'时，P=(F-1)/2，w2=w1.
'''
def AlexNet(input_shape = (224, 224, 3), output_shape = 2):

    model = Sequential()  #调用Keras中的sequential类创建模型对象

    model.add(Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                     activation='relu', input_shape=input_shape))   #输出为55*55*48
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2), padding='valid'))  #输出为27*27*48

    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     activation='relu'))    #输出为27*27*128
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))  #输出为13*13*128

    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='relu'))   #输出为13*13*192

    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='relu'))  # 输出为13*13*192

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='relu'))  # 输出为13*13*128
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))  # 输出为6*6*128

    model.add(Flatten())  #输出为4608的向量
    model.add(Dense(2304, activation='relu'))   #输出为2304个结点
    model.add(Dropout(0.3))

    model.add(Dense(2304, activation='relu'))  # 输出为2304个结点
    model.add(Dropout(0.3))

    model.add(Dense(2304, activation='softmax'))  # 输出分类结果

    return model
