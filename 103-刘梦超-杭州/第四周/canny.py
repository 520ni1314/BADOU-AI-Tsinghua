#!/usr/bin/env python 
# coding:utf-8
import cv2

# 读入原图
img = cv2.imread("lenna.png")
# opencv读进来的图片通道排列是BGR,需要转为RGB
trans_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 调用canny接口
edge_img = cv2.Canny(trans_rgb, 0, 300)
cv2.imshow("edge_img", edge_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
