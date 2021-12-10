#!/usr/bin/env python
# encoding=gbk

'''
Canny��Ե��⣺�Ż��ĳ���
'''
import cv2
import numpy as np


'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
��Ҫ������
��һ����������Ҫ�����ԭͼ�񣬸�ͼ�����Ϊ��ͨ���ĻҶ�ͼ��
�ڶ����������ͺ���ֵ1��
�������������ͺ���ֵ2��
apertureSize����ѡ������Sobel���ӵĴ�С
'''


#
class image_canny:
    def __init__(self,X,lowThreshold,max_lowThreshold):#��ʼ��ʵ������
        self.X = X
        self.lowThreshold=lowThreshold
        self.max_lowThreshold=max_lowThreshold
        self.image_gray = self._image_gray()
        self.image_threshold = self._image_threshold()
        self.image_canny = self._image_canny()
        cv2.imshow('canny', self.image_canny)
        cv2.waitKey(0)

    def _image_gray(self):
        print(self.X)
        image_gray = cv2.cvtColor(self.X, cv2.COLOR_BGR2GRAY)  # ת����ɫͼ��Ϊ�Ҷ�ͼ
        return image_gray


    def _image_threshold(self):#��˹�˲�
        print(self.image_gray)
        image_threshold=cv2.GaussianBlur(self.image_gray,(3,3),0)
        return image_threshold

    def _image_canny(self):
        image_canny = cv2.Canny(self.image_threshold,self.lowThreshold,self.max_lowThreshold)  # ��һ����������Ҫ�����ԭͼ�񣬸�ͼ�����Ϊ��ͨ���ĻҶ�ͼ���ڶ����������ͺ���ֵ1���������������ͺ���ֵ2��
        return image_canny



if __name__ == '__main__':
    img_source=cv2.imread('lenna.png')
    print(img_source)
    image_canny = image_canny(img_source,0,100)
