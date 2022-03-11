from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization


def AlexNet(input_shape=(224,224,3),output_shape=2):
    model = Sequential()
    # 使用步长为4，大小为11的卷积核进行卷积，输出特征层为96层
    # 为使运算速度加快这里将每个卷积filter减半
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11,11),
            strides=(4,4),
            padding='valid',
            input_shape=input_shape,
            activation='relu',
        )
    )
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
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
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'  # 如果padding设置为SAME，则说明输入图片大小和输出图片大小是一致的，如果是VALID则图片经过滤波器后可能会变小。
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
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )
    model.add(Flatten())   # 拍平
    # 减小模型大小原9216，改为1024
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape,activation='softmax'))

    return model

if __name__ == '__main__':
    model = AlexNet(output_shape=2)
    model.summary()