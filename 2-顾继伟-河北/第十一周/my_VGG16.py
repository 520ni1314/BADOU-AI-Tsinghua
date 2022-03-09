import tensorflow as tf

slim = tf.contrib.slim

def my_VGG16(
        inputs,
        num_classes=1000,
        is_Training=True,
        drop_keep_prob=0.5,
        spatial_squeeze=True,
        scope="vgg16"
        ):

    with tf.variable_scope(scope,'vgg16',[inputs]):
        net = slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')
        net = slim.max_pool2d(net,[2,2],scope='pool1')

        net = slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
        net = slim.max_pool2d(net,[2,2],scope='pool2')

        net = slim.repeat(net,2,slim.conv2d,256,[3,3],scope='conv3')
        net = slim.max_pool2d(net,[2,2],scope='pool3')

        net = slim.repeat(net,2,slim.conv2d,512,[3,3],scope='conv4')
        net = slim.max_pool2d(net,[2,2],scope='pool4')

        net = slim.repeat(net,2,slim.conv2d,512,[3,3],scope='conv5')
        net = slim.max_pool2d(net,[2,2],scope='pool5')

        net = slim.conv2d(net,4096,[1,1],scope='fc7')
        net = slim.dropout(net,drop_keep_prob,is_Training=is_Training,scope='dropout7')

        net = slim.conv2d(net,num_classes,[1,1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8'
                          )

        if spatial_squeeze:
            net = tf.squeeze(net,[1,2],name='fc8/squeezed')
        return net
