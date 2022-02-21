# -- coding:utf-8 --
import numpy as np
def Normalization(x):
    result = [(float(i)-min(x))/(max(x) - min(x)) for i in x]
    return result

def Normalization(x):
    result = [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
    return result
#标准化 均值为0 方差为1

def z_score(x):
    x_mean = np.mean(x)
    var = sum([(float(i)-np.mean(x))**2 / len(x) for i in x])
    result = [(float(x)- x_mean)/var for i in x]
    return result