#####################
# guass噪声
#####################

import numpy as np
import cv2
from numpy import shape
import random

# guass噪声函数
def GaussianNoise(src,means,sigma,percetage):
    # 输入图片
    NoiseImg=src
    # 计算加入噪声的像素点个数 = 百分比 * 像素总数
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    # 对所有的像素点进行循环处理
    for i in range(NoiseNum):
        # randX-随机生成的行，randY-随机生成的列
        # 高斯噪声图片边缘不处理，故-1
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)

        # 此处在原有像素灰度值上加上随机数
        # NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)

        # 阈值判断,若灰度值小于0则强制为0，若灰度值大于255则强制为255
        # if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg

img = cv2.imread('lenna.png',0)

# 利用guass噪声函数对img加入guass噪声
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)
