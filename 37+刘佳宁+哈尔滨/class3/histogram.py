
############################
# 实现图片直方图均衡
# 1.灰度图 -> 直方图均衡化
# 2.RGB彩色图像 -> 直方图均衡化
############################

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取原图像及灰度图
# img -> 512*512*3 , img_gray -> 512*512
img = cv2.imread("lenna.png",1)
img_gray = cv2.imread("lenna_gray.jpg",0)
# print("lenna原图尺寸：",img.shape,"lenna灰度图尺寸：",img_gray.shape)
# 显示lenna原图及灰度图
cv2.imshow("lenna", img)
cv2.waitKey(0)
cv2.imshow("lenna_gray",img_gray)
cv2.waitKey(0)

######################################################
# 对灰度图进行直方图均衡化
######################################################

# 利用cv2.equalizeHist得到灰度图的直方图均衡化后的图片img_gray_hist,512*512 -> 512*512
img_gray_hist = cv2.equalizeHist(img_gray)
# print("直方图均衡化后图片尺寸：",img_gray_hist.shape)
# 显示直方图均衡化后的图片并保存
cv2.imshow("lenna_gray_hist", img_gray_hist)
cv2.waitKey(0)
plt.imsave('lenna_gray_hist.jpg',img_gray_hist)

# 利用cv2.calcHist得到直方图（共有256个像素的位置），256*1
hist = cv2.calcHist([img_gray_hist],[0],None,[256],[0,256])
# print("直方图尺寸：",hist.shape)

# 输出直方图
plt.figure()
plt.title("lenna_gray Histogram")
plt.xlabel("pixel value")
plt.ylabel("numbers of pixel")
plt.plot(hist)
plt.xlim([0,256])
plt.show()



######################################################
# 对RGB彩色图像进行直方图均衡化
# 利用cv2.split和cv2.merge对RGB图片进行拆分和合并
######################################################

# 利用cv2.split对原图进行拆分，得到各个通道的图片，512*512*3 -> 512*512,512*512,512*512
(B,G,R) = cv2.split(img)

# 分别对每个通道进行直方图均衡化
img_B_hist = cv2.equalizeHist(B)
img_G_hist = cv2.equalizeHist(G)
img_R_hist = cv2.equalizeHist(R)

# 分别得到每个通道的直方图
B_hist = cv2.calcHist([img_B_hist],[0],None,[256],[0,256])
G_hist = cv2.calcHist([img_G_hist],[0],None,[256],[0,256])
R_hist = cv2.calcHist([img_R_hist],[0],None,[256],[0,256])

# 将直方图均衡化后的三个图片重新合并，得到最终图片
img_hist = cv2.merge((img_B_hist,img_G_hist,img_R_hist))

# 显示直方图均衡化后的图片并保存
cv2.imshow("lenna_hist", img_hist)
cv2.waitKey(0)
plt.imsave('lenna_hist.jpg',img_hist)