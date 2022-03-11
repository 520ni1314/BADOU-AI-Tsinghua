# Resnet网络

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import Activation,BatchNormalization,Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras import Model,Sequential

def identity_block(input_tensor,kernel_size,filters,stage,block):
    filters1,filters2,filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branh'

    x = Conv2D(filters=filters1,kernel_size=(1,1),name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters2,kernel_size=kernel_size,padding='same',name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)


    x = Conv2D(filters=filters3, kernel_size=(1, 1),name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x,input_tensor])
    x =Activation('relu')(x)

    return x

def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):
    filters1,filters2,filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters=filters1,kernel_size=(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters2, kernel_size=kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters=filters3,kernel_size=(1,1),strides=strides,name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)

    return x


def Resnet50(input_shape=[224,224,3],classes=1000):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3,3))(img_input)

    x = Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

    x = conv_block(x,3,[64,64,256],stage=2,block='a',strides=(1,1))
    x = identity_block(x,3,[64,64,256],stage=2,block='b')
    x = identity_block(x,3,[64,64,256],stage=2,block='c')

    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7,7),name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes,activation='relu',name='fc1000')(x)

    model = Model(img_input,x,name='resnet50')

    return model


if __name__ == '__main__':
    model = Resnet50()
    model.summary()

# 替换Resnet最后一层
    layer_fc = Dense(2,activation='softmax',name='fc2')(model.layers[-1].output)
    model2 = Model(model.input,layer_fc)
    model2.summary()
