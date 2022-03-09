import matplotlib as mpimg
import numpy as np
import tensorflow as tf
import cv2
import tensorflow.python.ops as array_ops

def load_image(path):
    img = mpimg.imread(path)
    short_edge = min(img.shape[:2])
    y = int((img.shape[0]-short_edge)/2)
    x = int((img.shape[1]-short_edge)/2)
    cut_img = img[y:y+short_edge,x:x+short_edge]

    return cut_img

def resize_image(img,size):
    with tf.name_scope("resize_image"):
        images = [];
        for i in img:
           i = cv2.resize(img,size)
           images.append(i)
        images = np.array(images)       #再把原图片转换成数组

        return images

def print_result(argmax):
    with open("./data/model/index_word.txt","r",encoding="utf-8") as f:
        result = [l.split(';')[1][:-1] for l in f.readlines()]
    return result[argmax]