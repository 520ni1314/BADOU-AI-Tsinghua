#!/usr/bin/env python
# encoding=gbk

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris

x,y=load_iris(return_X_y=True) #�������ݣ�x��ʾ���ݼ��е��������ݣ�y��ʾ���ݱ�ǩ
pca=dp.PCA(n_components=2) #����pca�㷨�����ý�ά�����ɷ���ĿΪ2
reduced_x=pca.fit_transform(x) #��ԭʼ���ݽ��н�ά��������reduced_x��
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): #���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı���
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
