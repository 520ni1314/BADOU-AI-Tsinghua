import numpy as np
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

"""
用卷积神经网络处理一组彩色图片时，Caffe/Theano 使用的数据格式是channels_first即：
（样本数，通道数，行数（高），列数（宽））
Tensforflow 使用的数据格式是channels_last即：
（样本数，行数（高），列数（宽），通道数）
"""
K.image_data_format() == 'channels_first'

if __name__ == '__main__':
    #初始化AlexNet对象
    model = AlexNet()
    #加载训练的权值
    weight_path = "D:\\个人\\AI精品班\\10_八斗清华班\\【11】图像识别\\代码\\AlexNet-Keras-master\\logs\\last1.h5"
    model.load_weights(weight_path)

    #加载需要预测的图像
    img = cv2.imread('./test2.jpg')
    #BGR->RGB
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #归一化
    img_nor = img_RGB/255

    #图像大小转为224*224
    img_resize = cv2.resize(img_nor,(224,224))
    img_resize = np.array(img_resize)
    # 图像维度提升为1*chanel*Weight*Height
    img_resize = np.expand_dims(img_resize, axis=0)

    print('the answer is: ', np.argmax(model.predict(img_resize)))
    cv2.imshow("ooo", img)
    cv2.waitKey(0)


