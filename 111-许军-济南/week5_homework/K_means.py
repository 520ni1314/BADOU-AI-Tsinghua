# -- coding:utf-8 --
import cv2
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time
def K_means(img,k,epochs,epsilon):
    rows,cols = img.shape[:]
    print(rows)
    data=img.reshape((rows * cols, 1))
    dst = np.zeros((rows * cols, 1))
    centers = {"{}".format(i):[random.randint(0,255)] for i in range(k)}
    for i in range(epochs):
        close_points = defaultdict(list)
        for j in data:
            close_c,close_dis = min([(k,(j-centers[k])**2) for k in centers],key=lambda t:t[1])
            close_points[close_c].append(j)
        for c in close_points:
            former_center = centers[c]
            neigbors_belong_to_c = close_points[c]
            neighbors_center=np.mean(neigbors_belong_to_c, axis=0)
            if abs(neighbors_center-former_center) > epsilon :
                centers[c]=neighbors_center  # 赋值新的中心点
        # 分割
    min_0 = min(close_points["0"])
    min_1 = min(close_points["1"])
    min_2 = min(close_points["2"])
    min_3 = min(close_points["3"])

    max_0 = max(close_points["0"])
    max_1 = max(close_points["1"])
    max_2 = max(close_points["2"])
    max_3 = max(close_points["3"])
    for i in range(len(data)):
        print(data[i])
        if data[i] >= min_0[0] and data[i] <= max_0[0]:
            dst[i] = 0
        elif data[i] >= min_1[0] and data[i] <= max_1[0]:
            dst[i] = 1
        elif data[i] >= min_2[0] and data[i] <= max_2[0]:
            dst[i] = 2
        else:
            dst[i]= 3
    m = dst.reshape((img.shape[0],img.shape[1]))
    return m


if __name__ == '__main__':
    img = cv2.imread("./img/lenna.png",0)
    img1 = K_means(img,4,10,1.0)
    plt.subplot(1,2,1)
    plt.imshow(img,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(img1,cmap="gray")
    plt.show()
