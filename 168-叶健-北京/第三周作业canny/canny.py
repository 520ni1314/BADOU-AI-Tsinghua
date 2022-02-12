#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
��Ҫ������
��һ����������Ҫ�����ԭͼ�񣬸�ͼ�����Ϊ��ͨ���ĻҶ�ͼ��
�ڶ����������ͺ���ֵ1��
�������������ͺ���ֵ2��
'''

img = cv2.imread("p3.jpg", 1)
# img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("grey", gray)
cv2.imshow("canny1", cv2.Canny(gray, 150, 300))
cv2.imshow("canny3", cv2.Canny(gray, 200, 400))
cv2.imshow("canny4", cv2.Canny(gray, 400, 500))
cv2.waitKey()
cv2.destroyAllWindows()
