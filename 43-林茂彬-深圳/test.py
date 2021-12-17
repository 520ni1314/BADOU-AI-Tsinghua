import cv2
import numpy as np


img = cv2.imread('lenna.png',0)



#获取图像高度、宽度
rows, cols = img.shape[:]
data = img.reshape((rows * cols, 1))
data = np.float32(data)
print(data)


