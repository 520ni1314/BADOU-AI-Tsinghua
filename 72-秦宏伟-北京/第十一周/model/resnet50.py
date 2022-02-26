from keras import models
from keras import layers
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
"""
Resnet50推理
Resnet Conv block，输入输出tensor的shape会改变
输入张量 input_tensor
核大小 kernel_size
特征数 filter
stage序号 stage
块名字 block
"""
def Conv_block(input_tensor,kernel_size,filter,stage,block,strides=(2,2)):
    filters1, filters2, filters3 = filter

    #卷积块名
    conv_name_base = 'res'+str(stage)+block+'_branch'
    #批标准化名
    bn_name_base = 'bn'+str(stage)+block+'_branch'

    x = layers.Conv2D(filters1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def identity_block(input_tensor,kernel_size,filter,stage,block):
    filters1, filters2, filters3 = filter

    # 卷积块名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 批标准化名
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1,(1,1),name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def ResNet50(input_shape=[224,224,3],classes=1000):

    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D((3, 3))(img_input)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = Conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = Conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = Conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

    model = models.Model(img_input, x, name='resnet50')

    weight_path = "D:\\个人\\AI精品班\\10_八斗清华班\\【11】图像识别\\代码\\resnet50_tf\\resnet50_weights_tf_dim_ordering_tf_kernels.h5"
    model.load_weights(weight_path)

    return model

if __name__ == '__main__':
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = ResNet50()
    #打印Resnet网络信息
    model.summary()
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))