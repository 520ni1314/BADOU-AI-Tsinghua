
##################################
# 图像的灰度化
##################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("lenna.png")
# 读取图像尺寸 512*512
h,w = img.shape[:2]
# 创建一个空白的与img同尺寸图像img_gray
img_gray = np.zeros([h,w],img.dtype)


###############################################
# 进行for双循环
# 1.找出h*w组3维BGR像素值
# 2.通过gray=0.11*B+0.59*G+0.3R得到像素点的灰度值
# 3.将各个像素点灰度值输入到img_gray中
###############################################
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

plt.imshow(img_gray)
plt.show()
plt.imsave('lenna_gray.jpg',img_gray)