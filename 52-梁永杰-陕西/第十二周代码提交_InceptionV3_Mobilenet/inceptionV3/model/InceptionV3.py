import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation,Dense,Input,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D


def conv2_bn(inputs,filters,kernel_size,strides=(1,1),padding='same',name=None):
    '''
    实现卷积--标准化--激活操作
    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :param padding:
    :param name:
    :return:
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(
        filters=filters,
        kernel_size=(kernel_size),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name
    )(inputs)
    x = BatchNormalization(scale=False,name=bn_name)(x)
    x = Activation('relu',name=name)(x)

    return x

def InceptionV3(input_shape=[299,299,3],classes=1000):
    img_imput = Input(shape=input_shape)

    x = conv2_bn(img_imput,32,(3,3),strides=(2,2),padding='valid')
    x = conv2_bn(x,32,(3,3),padding='valid')
    x = conv2_bn(x, 64, (3, 3))
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    x = conv2_bn(x, 80, (1, 1),padding='valid')
    x = conv2_bn(x, 192, (3, 3),padding='valid')
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    # --------------------------------- #
    # Inception Module1 35*35
    # --------------------------------- #
    # Inception Module1 part1
    # 35*35*192 -> 35*35*256
    branch1x1 = conv2_bn(x,64,(1,1))

    branch5x5 = conv2_bn(x,48,(1,1))
    branch5x5 = conv2_bn(branch5x5,64,(5,5))

    branch3x3dbl = conv2_bn(x,64,(1,1))
    branch3x3dbl = conv2_bn(branch3x3dbl,96,(3,3))
    branch3x3dbl = conv2_bn(branch3x3dbl,96,(3,3))

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2_bn(branch_pool,32,(1,1))

    # cancat输出通道数64+64+96+32=256
    x = layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3,name='mixed0')

    # Inception Module1 part2
    # 35*35*256 -> 35*35*288
    branch1x1 = conv2_bn(x, 64, (1, 1))

    branch5x5 = conv2_bn(x, 48, (1, 1))
    branch5x5 = conv2_bn(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2_bn(x, 64, (1, 1))
    branch3x3dbl = conv2_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2_bn(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2_bn(branch_pool, 64, (1, 1))

    # cancat输出通道数64+64+96+64=256
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed1')

    # Inception Module1 part3
    # 35*35*288 -> 35*35*288
    branch1x1 = conv2_bn(x, 64, (1, 1))

    branch5x5 = conv2_bn(x, 48, (1, 1))
    branch5x5 = conv2_bn(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2_bn(x, 64, (1, 1))
    branch3x3dbl = conv2_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2_bn(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2_bn(branch_pool, 64, (1, 1))

    # cancat输出通道数64+64+96+64=256
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed2')

    # -------------------------------------- #
    #     Inception Module2 17x17
    # -------------------------------------- #
    #  Inception Module2 part1
    # 35*35*288 -> 17*17*768
    branch3x3 = conv2_bn(x,384,(3,3),strides=(2,2),padding='valid')

    branch3x3dbl = conv2_bn(x,64,(1,1))
    branch3x3dbl = conv2_bn(branch3x3dbl,96,(3,3))
    branch3x3dbl = conv2_bn(branch3x3dbl,96,(3,3),strides=(2,2),padding='valid')

    branch_pool = MaxPooling2D((3,3),strides=(2,2))(x)

    # cancat输出通道数384+96+288=768
    x = layers.concatenate([branch3x3,branch3x3dbl,branch_pool],axis=3,name='mixed3')

    # Inception Module2 part2
    # 17*17*768 -> 17*17*768
    branch1x1 = conv2_bn(x,192,(1,1))

    branch7x7 = conv2_bn(x,128,(1,1))
    branch7x7 = conv2_bn(branch7x7,128,(1,7))
    branch7x7 = conv2_bn(branch7x7,192,(7,1))

    branch7x7dbl = conv2_bn(x,128,(1,1))
    branch7x7dbl = conv2_bn(branch7x7dbl,128,(7,1))
    branch7x7dbl = conv2_bn(branch7x7dbl,128,(1,7))
    branch7x7dbl = conv2_bn(branch7x7dbl,128,(7,1))
    branch7x7dbl = conv2_bn(branch7x7dbl,192,(1,7))

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2_bn(branch_pool,192,(1,1))
    x = layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3,name='mixed4')

    # Inception Module2 part3 and part4
    # 17*17*768 -> 17*17*768 -> 17*17*768
    for i in range(2):
        branch1x1 = conv2_bn(x,192,(1,1))

        branch7x7 = conv2_bn(x,160,(1,1))
        branch7x7 = conv2_bn(branch7x7,160,(1,7))
        branch7x7 = conv2_bn(branch7x7,192,(7,1))

        branch7x7dbl = conv2_bn(x,160,(1,1))
        branch7x7dbl = conv2_bn(branch7x7dbl,160,(7,1))
        branch7x7dbl = conv2_bn(branch7x7dbl,160,(1,7))
        branch7x7dbl = conv2_bn(branch7x7dbl,160,(7,1))
        branch7x7dbl = conv2_bn(branch7x7dbl,192,(1,7))

        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv2_bn(branch_pool,192,(1,1))
        x = layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3,name='mixed'+str(5+i))

    # Inception Module2 part5
    # 17*17*768 -> 17*17*768
    branch1x1 = conv2_bn(x,192,(1,1))

    branch7x7 = conv2_bn(x,192,(1,1))
    branch7x7 = conv2_bn(branch7x7,192,(1,7))
    branch7x7 = conv2_bn(branch7x7,192,(7,1))

    branch7x7dbl = conv2_bn(x,192,(1,1))
    branch7x7dbl = conv2_bn(branch7x7dbl,192,(7,1))
    branch7x7dbl = conv2_bn(branch7x7dbl,192,(1,7))
    branch7x7dbl = conv2_bn(branch7x7dbl,192,(7,1))
    branch7x7dbl = conv2_bn(branch7x7dbl,192,(1,7))

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2_bn(branch_pool,192,(1,1))
    x = layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3,name='mixed7')

    # -------------------------------------- #
    #     Inception Module3 8*8
    # -------------------------------------- #
    #  Inception Module3 part1
    # 35*35*768 -> 8*8*1280
    branch3x3 = conv2_bn(x,192,(1,1))
    branch3x3 = conv2_bn(branch3x3,320,(3,3),strides=(2,2),padding='valid')

    branch7x7x3 = conv2_bn(x,192,(1,1))
    branch7x7x3 = conv2_bn(branch7x7x3,192,(1,7))
    branch7x7x3 = conv2_bn(branch7x7x3,192,(7,1))
    branch7x7x3 = conv2_bn(branch7x7x3,192,(3,3),strides=(2,2),padding='valid')

    branch_pool =MaxPooling2D((3,3),strides=(2,2))(x)
    x = layers.concatenate([branch3x3,branch7x7x3,branch_pool],axis=3,name='mixed8')

    # Inception Module3 part2 and part3
    for i in range(2):
        branch1x1 = conv2_bn(x,320,(1,1))

        # 并联分支结构
        branch3x3 = conv2_bn(x,384,(1,1))
        branch3x3_1 = conv2_bn(branch3x3,384,(1,3))
        branch3x3_2 = conv2_bn(branch3x3,384,(3,1))
        branch3x3 = layers.concatenate([branch3x3_1,branch3x3_2],axis=3,name='mixed9_'+str(i))

        branch3x3dbl = conv2_bn(x,448,(1,1))
        branch3x3dbl = conv2_bn(branch3x3dbl,384,(3,3))
        branch3x3dbl_1 = conv2_bn(branch3x3dbl,384,(1,3))
        branch3x3dbl_2 = conv2_bn(branch3x3dbl,384,(3,1))
        branch3x3dbl = layers.concatenate([branch3x3dbl_1,branch3x3dbl_2],axis=3)

        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv2_bn(branch_pool,192,1,1)

        x = layers.concatenate([branch1x1,branch3x3,branch3x3dbl,branch_pool],axis=3,name='mixed'+str(9+i))

    # 平均池化后全连接
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes,activation='softmax',name='predictions')(x)

    inputs = img_imput

    model = Model(inputs,x,name='Inception_V3')

    return model


if __name__ == '__main__':
    model = InceptionV3()
    model.summary()




