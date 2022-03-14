from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization,MaxPooling2D,Flatten,Dense,Dropout

def MyAlexNet(input_shape=(224,224,3), output_size=2):
    model = Sequential()
    model.add(
        Conv2D(filters=48,
                kernel_size=(11,11),
               strides=(4,4),
               padding='valid',
               input_shape=input_shape,
               activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )
    model.add(
        Conv2D(filters=128,
               kernel_size=(5,5),
               strides=(1,1),
               padding='same',
               activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )
    model.add(
        Conv2D(filters=192,
               kernel_size=(3,3),
               strides=(1,1),
               padding='same',
               activation='relu'
        )
    )
    model.add(
        Conv2D(filters=192,
               kernel_size=(3,3),
               strides=(1,1),
               padding='same',
               activation='relu'
        )
    )
    model.add(
        Conv2D(filters=128,
               kernel_size=(3,3),
               strides=(1,1),
               padding='same',
               activation='relu'
        )
    )
    model.add(MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_size,activation='softmax'))

    return model
