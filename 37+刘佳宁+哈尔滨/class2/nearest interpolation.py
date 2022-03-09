
###########################
# 最近邻插值,512*512 -> 800*800
###########################

import cv2
import numpy as np
import matplotlib.pyplot as plt

def nearest(img):
    # 读取图像尺寸h,w,c
    h,w,c = img.shape
    # 创建800*800空图像emptyImage
    emptyImage = np.zeros((800,800,c),np.uint8)
    scale_h = 800/h
    scale_w = 800/w
    ###############
    # for循环
    # 最近邻插值操作
    ###############
    for i in range(800):
        for j in range(800):
            x = int(i/scale_h)
            y = int(j/scale_w)
            emptyImage[i,j] = img[x,y]
    return emptyImage

# 读取图像
img = cv2.imread("lenna.png")
# 最近邻插值操作
image_nearest = nearest(img)
cv2.imshow("image_nearest",image_nearest)
cv2.waitKey(0)
plt.imsave('lenna_nearest.jpg',image_nearest)

