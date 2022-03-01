#!/usr/bin/env python
# encoding=gbk

import matplotlib.pyplot as plt
from sklearn.datasets._base import load_iris
import numpy as np


class DimReduction:
    """PCA降维过程实现"""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.zero_center_mat = np.zeros(self.data.shape)
        self.cov_matrix = np.array((self.data.ndim, self.data.ndim))
        self.eigenvalue = np.array(self.data.ndim)
        self.eigenvector = np.array((self.data.ndim, self.data.ndim))
        self.trans_mat = np.array((self.data.ndim, self.data.ndim))
        self.sorted_eigenvalue_ind = None
        self.dimension_k = -1
        self.matrix_k = None

    def zero_centered(self):
        """对样本矩阵进行0均值化(亦称作中心化)"""
        average = [np.average(x) for x in self.data.T]
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.zero_center_mat[i, j] = self.data[i, j] - average[j]
        print(self.zero_center_mat)

    def cov_mat(self):
        """求协方差矩阵"""
        self.cov_matrix = np.mat(self.zero_center_mat.T) * np.mat(self.zero_center_mat) / (self.data.shape[0] - 1)
        print("数据协方差矩阵为：\n")
        print(self.cov_matrix)

    def eigen(self):
        """求协方差矩阵的特征值和特征向量，协方差矩阵经特征向量矩阵的正交变换作用后成为对角矩阵，其主对角线元素由对应的特征值排列组成"""
        self.eigenvalue, self.eigenvector = np.linalg.eig(self.cov_matrix)
        print(f"协方差矩阵的特征值：{self.eigenvalue}\n特征向量(是列向量)：{self.eigenvector}")

    def eigen_sort(self):
        """对特征向量矩阵进行初等行变换，目的为将对角矩阵主对角线元素进行非递减排序"""
        self.sorted_eigenvalue_ind = np.argsort(-self.eigenvalue)
        self.trans_mat = np.mat([np.array(self.eigenvector)[:, m] for m in self.sorted_eigenvalue_ind])
        print(f"特征值排序：{self.sorted_eigenvalue_ind}\n过渡矩阵：{self.trans_mat}")

    def dimension_reduce(self, info_res_ratio=0.98):
        """根据输入降维后矩阵保有率，计算降维后矩阵的维数dimension_k"""
        eigenvalue_sum = 0
        while eigenvalue_sum / sum(self.eigenvalue) < info_res_ratio:
            eigenvalue_sum += self.eigenvalue[self.sorted_eigenvalue_ind[self.dimension_k + 1]]
            self.dimension_k += 1
        print(f"满足保留{info_res_ratio * 100}%的原始信息的降维矩阵是{self.dimension_k + 1}维。")

    def k_matrix(self):
        """生成降维矩阵matrix_k"""
        self.matrix_k = np.mat(
            [np.array(self.eigenvector)[:, n] for n in self.sorted_eigenvalue_ind[:self.dimension_k + 1]])
        print(f"降维后的变换矩阵为：{self.matrix_k.T}")

    def new_array(self):
        """生成降维后矩阵并返回"""
        return np.mat(self.data) * self.matrix_k.T


in_data, in_label = load_iris(return_X_y=True)
iris_l = DimReduction(in_data, in_label)
iris_l.zero_centered()
iris_l.cov_mat()
iris_l.eigen()
iris_l.eigen_sort()
iris_l.dimension_reduce(0.95)
iris_l.k_matrix()
new_iris_l = np.array(iris_l.new_array())

# 对数据分类的可视化
red_data, red_label = [], []
blue_data, blue_label = [], []
green_data, green_label = [], []
# 按鸢尾花的类别将降维后的数据点保存在不同的标签中
for i in range(len(new_iris_l)):
    if iris_l.labels[i] == 0:
        red_data.append(new_iris_l[i][0])
        red_label.append(new_iris_l[i][1])
    elif iris_l.labels[i] == 1:
        blue_data.append(new_iris_l[i][0])
        blue_label.append(new_iris_l[i][1])
    else:
        green_data.append(new_iris_l[i][0])
        green_label.append(new_iris_l[i][1])
# 画图
plt.scatter(red_data, red_label, c='r', marker='x', label="red")
plt.scatter(blue_data, blue_label, c='b', marker='D', label='blue')
plt.scatter(green_data, green_label, c='g', marker='.', label='green')
plt.title("Classification of iris L.")
plt.legend()
plt.show()
