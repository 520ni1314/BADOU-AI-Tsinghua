from keras import models
from keras import layers

"""
定义VGG16网络结构
"""
def VGG_16(input_shape=(224,224,3),output_shape=2):
    model = models.Sequential()
    # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)，再2X2最大池化，输出net为(112,112,64)。
    # 224*224*3 -> 224*224*64
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape,
                            activation='relu'))
    model.add(layers.BatchNormalization())
    # 224*224*64 -> 224*224*64
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 224*224*64 ->112*112*64
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

    # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)，再2X2最大池化，输出net为(56,56,128)。
    # 112*112*64 -> 112*112*128
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 112*112*128 -> 112*112*128
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 112*112*128 -> 56*56*128
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

    #conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)，再2X2最大池化，输出net为(28,28,256)。
    # 56*56*128 -> 56*56*256
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 56*56*256 -> 56*56*256
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 56*56*256 -> 56*56*256
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 56*56*256 ->28*28*256
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

    #conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(28,28,512)，再2X2最大池化，输出net为(14,14,512)。
    # 28*28*256 -> 28*28*512
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 28*28*512 -> 28*28*512
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 28*28*512 -> 28*28*512
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 28*28*512 ->14*14*512
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

    #conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(14,14,512)，再2X2最大池化，输出net为(7,7,512)。
    # 14*14*512 -> 14*14*512
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 14*14*512 -> 14*14*512
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 14*14*512 -> 14*14*512
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # 14*14*512 -> 7*7*512
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='valid'))

    # # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)。共进行两次。
    model.add(layers.Conv2D(filters=4096, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu'))
    model.add(layers.Dropout(0.5))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=4096, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'))
    model.add(layers.Dropout(0.5))

    # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)。
    # model.add(layers.Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_shape, activation='softmax'))
    # 基于全连接网络-----------------------------
    # model.add(layers.Flatten())
    # model.add(layers.Dense(4096, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(4096, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(output_shape, activation='softmax'))
    # 基于全连接网络-----------------------------

    return model