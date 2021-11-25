# -*- coding: utf-8 -*-
"""
2021-11-24 彩色图像的灰度化
"""


from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
def togray(img):
    img_source = cv2.imread("lenna.png") # 导入图片
#print(img_gray.shape) #得到图片的（行，列，通道）
    img_source_h,img_source_w=img_source.shape[0:2]
    img_gray = np.zeros([img_source_h,img_source_w],img_source.dtype)   #创建一张和当前图片大小一样的单通道图片 np.zeros默认dtype 为float 需要指定跟图片一致的dtype


    for i in range(img_source_h): #循环获取每个坐标
        for j in range(img_source_w):
            a=img_source[i,j]
            img_gray[i, j] = int(a[0] * 0.11 + a[1] * 0.59 + a[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
    return img_gray

def tobinary(img):
    img_gray1 = rgb2gray(img)
    img_binary = np.where(img_gray1 <0.5, 0, 255).astype(np.uint8)
    return img_binary


if __name__ == '__main__':
    img_source = cv2.imread('lenna.png')
    img_gray = togray(img_source)
    img_binary = tobinary(img_source)
    cv2.imshow('img_source', img_source)
    cv2.imshow('img_gray', img_gray)
    cv2.imshow('img_binary', img_binary)
    cv2.waitKey(0)