#####################
# 椒盐噪声
#####################

import numpy as np
import cv2
from numpy import shape
import random

# 椒盐噪声函数
def fun(src,percetage):
    # 输入图片
    NoiseImg = src
    # 计算加入噪声的像素点个数 = 百分比 * 像素总数
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    # 对所有的像素点进行循环处理
    for i in range(NoiseNum):
        # randX-随机生成的行，randY-随机生成的列
        # 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)

        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

img = cv2.imread('lenna.png',0)

# 利用椒盐噪声函数对img加入椒盐噪声
img1 = fun(img,0.5)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('source',img2)
cv2.imshow('lenna_jiaoyan',img1)
cv2.waitKey(0)