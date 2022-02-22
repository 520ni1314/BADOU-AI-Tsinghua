#!/usr/bin/env python 
# coding:utf-8
import cv2
import numpy as np
import tensorflow as tf


# 调整图片的尺寸
def resize_img(inputs, shape):
    with tf.name_scope('resize_img'):
        img_list = []
        for img in inputs:
            new_img = cv2.resize(img, shape)
            img_list.append(new_img)
        # 转为多维数组
        images = np.array(img_list)
        return images


def print_answer(argmax):
    with open("/Users/lmc/Documents/workFile/AI/week11/test/AlexNet-Keras/data/model/index_word.txt", "r",
              encoding='utf-8') as f:
        # 去除空格
        labels = [l.split(";")[1].strip() for l in f.readlines()]

    return labels[argmax]
