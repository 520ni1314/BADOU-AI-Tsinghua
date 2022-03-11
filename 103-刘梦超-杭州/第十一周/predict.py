#!/usr/bin/env python 
# coding:utf-8
import cv2
import numpy as np
from AlexNetModel import AlexNet
import utils

if __name__ == '__main__':
    # 加载图像
    img = cv2.imread("/Users/lmc/Documents/workFile/AI/week11/test/AlexNet-Keras/test2.jpg")
    img_nor = img / 255
    # 增加一个维度
    img_nor = np.expand_dims(img_nor, axis=0)
    # 调整图像的尺寸
    img_nor = utils.resize_img(img_nor, (224, 224))
    # 加载模型
    model = AlexNet()
    # 加载权重
    model.load_weights("/Users/lmc/Documents/workFile/AI/week11/test/AlexNet-Keras/logs/last1.h5")
    # 预测
    predict = model.predict(img_nor)
    # 最大值索引
    argmax = np.argmax(predict)
    # 映射中文标签
    label = utils.print_answer(argmax)
    print("神经网络认为该图像是:", label)
    cv2.imshow("src", img)
    if cv2.waitKey(0) == 27:
        cv2.destroyWindow()
