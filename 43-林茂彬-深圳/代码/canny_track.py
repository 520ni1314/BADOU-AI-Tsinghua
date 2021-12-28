#!/usr/bin/env python
# encoding=gbk
 
'''
Canny��Ե��⣺�Ż��ĳ���
'''
import cv2
import numpy as np 
 
def CannyThreshold(lowThreshold):  
    detected_edges = cv2.GaussianBlur(gray,(3,3),0) #��˹�˲� 
    detected_edges = cv2.Canny(detected_edges,
            lowThreshold,
            lowThreshold*ratio,
            apertureSize = kernel_size)  #��Ե���
 
     # just add some colours to edges from original image.  
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  #��ԭʼ��ɫ��ӵ����ı�Ե��
    cv2.imshow('canny demo',dst)  
  
 
lowThreshold = 0  
max_lowThreshold = 100  
ratio = 3  
kernel_size = 3  
  
img = cv2.imread('lenna.png')  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #ת����ɫͼ��Ϊ�Ҷ�ͼ
  
cv2.namedWindow('canny demo')  
  
#���õ��ڸ�,
'''
�����ǵڶ���������cv2.createTrackbar()
����5����������ʵ������������������ʹ����֪����ʲô��˼��
��һ�������������trackbar���������
�ڶ��������������trackbar����������������
�����������������trackbar��Ĭ��ֵ,Ҳ�ǵ��ڵĶ���
���ĸ������������trackbar�ϵ��ڵķ�Χ(0~count)
������������ǵ���trackbarʱ���õĻص�������
'''
cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)  
  
CannyThreshold(0)  # initialization  
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()  
