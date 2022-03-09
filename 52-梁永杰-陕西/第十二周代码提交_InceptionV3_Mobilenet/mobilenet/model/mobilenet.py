from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D,Conv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras import backend as K



def conv_block(inputs,filters,kernel_size=(3,3),strides=(1,1)):
    '''
    普通卷积块，包含卷积层，标准化层以及激活层
    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :return:
    '''
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6,name='conv1_relu')(x)
    return x


def depthwise_conv_block(inputs,pointwise_conv_filters,depth_multiplier=1,
                         strides=(1,1),block_id=1):
    '''
    深度可分离卷积块，包含了Depthwise Conv和1x1加深深度的Pointwise Conv卷积
    :param inputs:
    :param pointwise_conv_filters:
    :param depth_multiplier:
    :param strides:
    :param block_id:
    :return:
    '''
    x = DepthwiseConv2D((3,3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6,name='conv_dw_%d_relu' % block_id)(x)


    x = Conv2D(filters=pointwise_conv_filters,
               kernel_size=(1,1),
               strides=(1,1),
               use_bias=False,
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6,name='conv_pw_%d_relu' % block_id)(x)
    return x




def relu6(x):
    return K.relu(x,max_value=6)


def Mobilenet(input_shape=[224,224,3],depth_multiplier=1,dropout=1e-3,classes=1000):
    img_input = Input(shape=input_shape)

    # 224*224*3 -> 112*112*32
    x = conv_block(img_input,filters=32,strides=(2,2))

    # 112*112*32 -> 112*112*64
    x = depthwise_conv_block(x,pointwise_conv_filters=64,
                             depth_multiplier=depth_multiplier,block_id=1)

    # 112*112*64 -> 56*56*128
    x = depthwise_conv_block(x,pointwise_conv_filters=128,
                             depth_multiplier=depth_multiplier,strides=(2,2),block_id=2)

    # 56*56*128 -> 56*56*128
    x = depthwise_conv_block(x, pointwise_conv_filters=128,
                             depth_multiplier=depth_multiplier, block_id=3)

    # 56*56*128 -> 28*28*256
    x = depthwise_conv_block(x, pointwise_conv_filters=256,
                             depth_multiplier=depth_multiplier,strides=(2,2), block_id=4)

    # 28*28*256 -> 28*28*256
    x = depthwise_conv_block(x, pointwise_conv_filters=256,
                             depth_multiplier=depth_multiplier, block_id=5)

    # 28*28*256 -> 14*14*512
    x = depthwise_conv_block(x, pointwise_conv_filters=512,
                             depth_multiplier=depth_multiplier, strides=(2, 2), block_id=6)

    # 14*14*512 -> 14*14*512
    x = depthwise_conv_block(x, pointwise_conv_filters=512,
                             depth_multiplier=depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, pointwise_conv_filters=512,
                             depth_multiplier=depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, pointwise_conv_filters=512,
                             depth_multiplier=depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, pointwise_conv_filters=512,
                             depth_multiplier=depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, pointwise_conv_filters=512,
                             depth_multiplier=depth_multiplier, block_id=11)

    # 14*14*512 -> 7*7*1024
    x = depthwise_conv_block(x, pointwise_conv_filters=1024,
                             depth_multiplier=depth_multiplier, strides=(2, 2), block_id=12)

    # 7*7*1024 -> 7*7*1024
    x = depthwise_conv_block(x, pointwise_conv_filters=1024,
                             depth_multiplier=depth_multiplier, block_id=13)

    # 7*7*1024 -> 1*1*1024
    x = GlobalAveragePooling2D()(x)   # or AveragePooling2D(Pooling=(7,7))
    x = Reshape((1,1,1024),name='reshape_1')(x)
    x = Dropout(dropout,name='dropout')(x)
    x = Conv2D(classes,(1,1),padding='same',name='conv_preds')(x)
    x = Activation('softmax',name='act_softmax')(x)
    x = Reshape((classes,),name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs,x,name='mobilenet_1_0_224_tf')

    return model

if __name__ == '__main__':
    model = Mobilenet()
    model.summary()
















