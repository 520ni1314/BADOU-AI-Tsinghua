#!/usr/bin/env python
# encoding=gbk

import matplotlib.pyplot as plt
from sklearn.datasets._base import load_iris
import numpy as np


class DimReduction:
    """PCA��ά����ʵ��"""

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
        """�������������0��ֵ��(��������Ļ�)"""
        average = [np.average(x) for x in self.data.T]
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.zero_center_mat[i, j] = self.data[i, j] - average[j]
        print(self.zero_center_mat)

    def cov_mat(self):
        """��Э�������"""
        self.cov_matrix = np.mat(self.zero_center_mat.T) * np.mat(self.zero_center_mat) / (self.data.shape[0] - 1)
        print("����Э�������Ϊ��\n")
        print(self.cov_matrix)

    def eigen(self):
        """��Э������������ֵ������������Э�������������������������任���ú��Ϊ�ԽǾ��������Խ���Ԫ���ɶ�Ӧ������ֵ�������"""
        self.eigenvalue, self.eigenvector = np.linalg.eig(self.cov_matrix)
        print(f"Э������������ֵ��{self.eigenvalue}\n��������(��������)��{self.eigenvector}")

    def eigen_sort(self):
        """����������������г����б任��Ŀ��Ϊ���ԽǾ������Խ���Ԫ�ؽ��зǵݼ�����"""
        self.sorted_eigenvalue_ind = np.argsort(-self.eigenvalue)
        self.trans_mat = np.mat([np.array(self.eigenvector)[:, m] for m in self.sorted_eigenvalue_ind])
        print(f"����ֵ����{self.sorted_eigenvalue_ind}\n���ɾ���{self.trans_mat}")

    def dimension_reduce(self, info_res_ratio=0.98):
        """�������뽵ά��������ʣ����㽵ά������ά��dimension_k"""
        eigenvalue_sum = 0
        while eigenvalue_sum / sum(self.eigenvalue) < info_res_ratio:
            eigenvalue_sum += self.eigenvalue[self.sorted_eigenvalue_ind[self.dimension_k + 1]]
            self.dimension_k += 1
        print(f"���㱣��{info_res_ratio * 100}%��ԭʼ��Ϣ�Ľ�ά������{self.dimension_k + 1}ά��")

    def k_matrix(self):
        """���ɽ�ά����matrix_k"""
        self.matrix_k = np.mat(
            [np.array(self.eigenvector)[:, n] for n in self.sorted_eigenvalue_ind[:self.dimension_k + 1]])
        print(f"��ά��ı任����Ϊ��{self.matrix_k.T}")

    def new_array(self):
        """���ɽ�ά����󲢷���"""
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

# �����ݷ���Ŀ��ӻ�
red_data, red_label = [], []
blue_data, blue_label = [], []
green_data, green_label = [], []
# ���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı�ǩ��
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
# ��ͼ
plt.scatter(red_data, red_label, c='r', marker='x', label="red")
plt.scatter(blue_data, blue_label, c='b', marker='D', label='blue')
plt.scatter(green_data, green_label, c='g', marker='.', label='green')
plt.title("Classification of iris L.")
plt.legend()
plt.show()
