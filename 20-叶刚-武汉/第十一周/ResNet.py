"""
基于tensorflow.keras的ResNet分类网络实现
tensorflow: 2.3.0
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, BatchNormalization, Activation


def ConvBlock(input_tensor, filters: tuple, stage: int, index: int, strides=(2, 2)):
    # ConvBlock的输入和输出维度不一致
    conv_base_name = 'ConvBlock' + str(stage) + '_' + str(index) + '_'
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, kernel_size=(1, 1), strides=(1, 1), name=conv_base_name + 'a')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size=(3, 3), strides=strides, padding='same', name=conv_base_name + 'b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size=(1, 1), strides=(1, 1), name=conv_base_name + 'c')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3, kernel_size=(1, 1), strides=strides, name=conv_base_name + 'shortcut')(input_tensor)

    out = layers.add([x, shortcut])
    out = Activation('relu')(out)
    return out


def IdentityBlock(input_tensor, filters: tuple, stage: int, index: int):
    # IdentityBlock的输入和输出维度一致
    identity_base_name = 'IdentityBlock' + str(stage) + '_' + str(index) + '_'
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, kernel_size=(1, 1), strides=(1, 1), name=identity_base_name + 'a')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=identity_base_name + 'b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size=(1, 1), strides=(1, 1), name=identity_base_name + 'c')(x)
    x = BatchNormalization()(x)

    out = layers.add([x, input_tensor])
    out = Activation('relu')(out)
    return out


def ResNet50(input_shape=(224, 224, 3), num_classes=1000):
    input_tensor = Input(shape=input_shape, name='Input')
    x = ZeroPadding2D(padding=(3, 3), name='ZeroPadding')(input_tensor)

    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='Conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    # 因为MaxPooling2D降采样了一次，紧接着的第一个ConvBlock的步长为(1,1)，其他的ConvBlock步长为(2,2)
    x = ConvBlock(x, (64, 64, 256), stage=2, index=1, strides=(1, 1))
    x = IdentityBlock(x, (64, 64, 256), stage=2, index=2)
    x = IdentityBlock(x, (64, 64, 256), stage=2, index=3)

    x = ConvBlock(x, (128, 128, 512), stage=3, index=1)
    x = IdentityBlock(x, (128, 128, 512), stage=3, index=2)
    x = IdentityBlock(x, (128, 128, 512), stage=3, index=3)
    x = IdentityBlock(x, (128, 128, 512), stage=3, index=4)

    x = ConvBlock(x, (256, 256, 1024), stage=4, index=1)
    x = IdentityBlock(x, (256, 256, 1024), stage=4, index=2)
    x = IdentityBlock(x, (256, 256, 1024), stage=4, index=3)
    x = IdentityBlock(x, (256, 256, 1024), stage=4, index=4)
    x = IdentityBlock(x, (256, 256, 1024), stage=4, index=5)
    x = IdentityBlock(x, (256, 256, 1024), stage=4, index=6)

    x = ConvBlock(x, (512, 512, 2048), stage=5, index=1)
    x = IdentityBlock(x, (512, 512, 2048), stage=5, index=2)
    x = IdentityBlock(x, (512, 512, 2048), stage=5, index=3)

    x = AveragePooling2D(pool_size=(7, 7), name='AveragePooling')(x)
    x = Flatten()(x)
    y = Dense(num_classes, name='fc')(x)  # activation='softmax', 此处不做激活，在损失函数里设置from_logits=True即可

    model = tf.keras.Model(input_tensor, y, name='ResNet50-V1.5')
    return model
