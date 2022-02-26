from keras import models
from keras import layers
"""
定义AlexNet神经网络
"""
def AlexNet(input_shape=(224,224,3),output_shape=2):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=48,kernel_size=(11,11),strides=(4,4),padding='valid',input_shape=input_shape,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))

    model.add(layers.Conv2D(filters=128,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))

    model.add(layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(output_shape,activation='softmax'))

    return model











