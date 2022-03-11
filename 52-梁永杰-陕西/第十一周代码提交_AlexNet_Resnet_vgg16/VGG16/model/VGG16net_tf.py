# tensorflow slim 实现VGG16网络结构
# slim是一个使构建，训练，评估神经网络变得简单的库。它可以消除原生tensorflow里面很多重
# 复的模板性的代码，让代码更紧凑，更具备可读性。另外slim提供了很多计算机视觉方面的著名模型
# （VGG, AlexNet等），我们不仅可以直接使用，甚至能以各种方式进行扩展。

import tensorflow as tf
from tensorflow.contrib import slim


def VGG16net(inputs,
             num_classes=2,
             is_training=True,
             dropout_keep_prob=0.5,
             spatial_squeeze=True,
             scope='vgg_16'):
    # 通过组合slim中变量(variables)、网络层(layer)、前缀名(scope)，
    # 模型可以被简洁的定义
    with tf.variable_scope(scope,'vgg_16',[inputs]):
        # VGG16
        # Slim也提供了两个元运算符----repeat和stack，
        # 允许用户可以重复地使用相同的运算符。
        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        net = slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')
        # 2X2最大池化，输出net为(112,112,64)
        net = slim.max_pool2d(net,[2,2],scope='pool1')
        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
        net = slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
        # 2X2最大池化，输出net为(56,56,128)
        net = slim.max_pool2d(net,[2,2],scope='pool2')
        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
        net = slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net,[2,2],scope='pool3')
        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
        net = slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv4')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net,[2,2],scope='pool4')
        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
        net = slim.repeat(net,3,256,slim.conv2d,[3,3],scope='conv5')
        # 2X2最大池化，输出net为(7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net,4096,[7,7],padding='VALID',scope='fc6')
        net = slim.dropout(net,dropout_keep_prob,is_training=is_training,
                           scope='dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net,4096,[1,1],scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net,num_classes,[1,1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        # 因采用卷积模拟全连接，故输出需平铺
        if spatial_squeeze:

            net = tf.squeeze(net,[1,2],name='fc8/squeezed')# 该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果axis可以用来指定要删掉的为1的维度，此处要注意指定的维度必须确保其是1，否则会报错
        return net
