#####################
# K-means聚类算法
#####################

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片信息,img -> 512*512*3
img = cv2.imread('lenna.png')
print(img.shape)

# 图像二维像素转换为一维,同时转为浮点数,512*512*3 -> 262144*3
data = img.reshape((-1,3))
data = np.float32(data)
print(data.shape)

# 停止条件(type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类,将lenna图像分为两类：人像+背景
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

#图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

#图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)

images = [img, dst2]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
plt.show()