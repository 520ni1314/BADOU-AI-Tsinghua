
####################
# 实现canny检测
####################

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

################
# canny边缘检测
################
def Canny(Threshold):
    # 利用cv2.GaussianBlur进行高斯滤波,kernel_size = (3,3)
    detected_guass = cv2.GaussianBlur(gray,(3,3),0)
    # Canny边缘检测
    detected_canny = cv2.Canny(detected_guass,
            Threshold,
            Threshold*ratio,
            apertureSize = kernel_size)

    dst = cv2.bitwise_and(img,img,mask = detected_canny)
    cv2.imshow('canny demo',dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

Threshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图

Canny(0)