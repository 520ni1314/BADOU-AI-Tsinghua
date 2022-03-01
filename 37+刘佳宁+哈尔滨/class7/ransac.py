#####################
# ransac思想
"""
  ransac思想步骤：
  1. 在数据中随机选择几个点设定为内群；
  2. 计算适合内群的模型，并计算参数；
  3. 把其他没选到的点带入刚刚建立的模型中，计算是否为内群（自己设定阈值）；
  4. 记下内群数量；
  5. 重复达到迭代次数；
  6.比较哪次计算中内群数量最多，即为所求模型。
”“”
#####################

import numpy as np
import scipy as sp
import scipy.linalg as sl

def ransac(data, model, n, k, t, d, debug = False, retur_all = False):






class LinearLeastSquareModel:
    #最小二乘求线性解,用于计算RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
		#np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B) #residues:残差和

        return x #返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi
        B_fit = sp.dot(A, model) #计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum( (B - B_fit) ** 2, axis = 1 ) #sum squared error per row

        return err_per_point