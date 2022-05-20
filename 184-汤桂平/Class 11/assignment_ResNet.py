# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/5/20

import numpy as np
from keras import layers
from keras.preprocessing import image

from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions


#利用函数式API创建残差网络
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2a')(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([shortcut, x])
    x = Activation('relu')(x)

    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2a')(x)

    x = layers.add([input_tensor, x])
    x = Activation('relu')(x)

    return x

def ResNet50(input_shape=(224, 224, 3), classes=1000):
    input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
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

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc')(x)

    model = Model(input, x, name='resnet50')
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")   #加载已有的网络权重
    return model

if __name__ == "__main__":
    model = ResNet50()
    model.summary()   #确认模型摘要
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))    #Loads an image into PIL format
    x = image.img_to_array(img)    #Converts a PIL Image instance to a Numpy array
    x = np.expand_dims(x, axis=0)
    '''
    Expand the shape of an array.
    Insert a new axis that will appear at the `axis` position in the expanded array shape.
    '''
    x = preprocess_input(x)    #Preprocesses a tensor or Numpy array encoding a batch of images.

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))