# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/5/10

import numpy as np
import cv2
import tensorflow as tf

def resize_image(images, size):
    with tf.name_scope('resize image'):
        images_resize = []
        for i in images:
            i = cv2.resize(i, size)
            images_resize.append(i)
        images_resize = np.array(images_resize)
    return images_resize

def print_answer(argmax):
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    # print(synset[argmax])
    return synset[argmax]