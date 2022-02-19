from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation, Softmax, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
# np_utils功能类似one-hot

#首先确定输入输出的维度
def my_AlexNet(input_shape=(224*224*3),output_shape=2):
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
    model.add(
        MaxPool2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )




    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(
        MaxPool2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )

    model.add(
        MaxPool2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )





    #全连接之前必要拍扁
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape,activation='softmax'))

    return model

