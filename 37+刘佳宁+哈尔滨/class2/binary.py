
##################################
# 图像的二值化
##################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 读取灰度化后的图像,0——读取灰度图，512*512；如果没有0，则读取RGB图像，512*512*3
img = cv2.imread("lenna_gray.jpg",0)
# 读取图像尺寸 h = 512,w = 512
h,w = img.shape[:2]

# 创建一个空白的与img同尺寸图像img_dinary,512*512
img_dinary = np.zeros([h,w],img.dtype)

# 设置阈值125
threshold = 125

####################################
# for双循环
# 1.用if判断每个像素点与阈值的大小
# 2.在img_dinary中更新像素值
####################################
for i in tqdm(range(h)):
    for j in range(w):
        if (img[i,j] < threshold):
            img_dinary[i,j] = 0
        else:
            img_dinary[i,j] = 1

plt.imshow(img_dinary)
plt.show()
plt.imsave('lenn_binary.jpg',img_dinary)