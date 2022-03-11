######################
# 实现PCA主成分分析
######################

import numpy as np
from matplotlib import pyplot as plt

#################################################
# PCA算法具体流程
# 1.去均值中心化
# 2.
#################################################
class PCA(object):
    # 对原始数据集X进行PCA降维，得到K阶降维矩阵Z,m*n -> m*K\

    # 初始化
    def __init__(self, X, K):
        # 初始数据集的样本矩阵X
        self.X = X
        # K阶降维矩阵，降维选择的特征数
        self.K = K
        # 矩阵X的中心化
        self.centrX = []
        # 数据样本集的协方差矩阵C,n*n
        self.C = []
        # 样本矩阵X的降维转换矩阵U,n*K
        self.U = []
        # 样本矩阵X的降维矩阵Z,m*K
        self.Z = []

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    # 计算样本矩阵X的中心化处理后的新矩阵centrX
    def _centralized(self):
        print('样本矩阵X:', self.X)
        # 设置空中心化
        centrX = []

        # 求得样本集的特征均值
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值:', mean)

        # 中心化处理，x = x - mean
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:', centrX)

        return centrX

    # 计算样本矩阵X的协方差矩阵C
    def _cov(self):
        # 计算样本集的样例个数m
        ns = np.shape(self.centrX)[0]

        # 计算样本矩阵X的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print('样本矩阵X的协方差矩阵C:', C)

        return C

    # 计算特征矩阵X的降维转换矩阵U,n*K,n是X的特征维度总数，k是降维矩阵的特征维度
    def _U(self):
        # 利用np.linalg.eig分别计算特征矩阵X的协方差矩阵C的特征值a和特征向量b
        a,b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:', a,'样本集的协方差矩阵C的特征向量:\n', b)

        # 利用np.argsort给出特征值降序的topK的索引序列ind,因为argsort是从小到大排序，所以a -> -1*a
        ind = np.argsort(-1*a)

        # 构建K阶降维的降维转换矩阵U,U = UT'
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:'%self.K, U)

        return U

    # 计算降维矩阵Z = XU,m*n·n*K -> m*K
    # n是样本总数,K是降维矩阵中特征维度总数
    def _Z(self):
        # 计算降维矩阵Z = XU
        Z = np.dot(self.X, self.U)
        print('样本矩阵X的降维矩阵Z:', Z)

        return Z

if __name__=='__main__':
    'X -> 10*3, 10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])

    # 设定K=2
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X,K)























