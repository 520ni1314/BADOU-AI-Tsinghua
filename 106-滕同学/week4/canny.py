# author: orea

import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
��Ҫ������
��һ����������Ҫ�����ԭͼ�񣬸�ͼ�����Ϊ��ͨ���ĻҶ�ͼ��
�ڶ����������ͺ���ֵ1��
�������������ͺ���ֵ2��
'''

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 80, 150)
cv2.imwrite("canny.png", canny)

