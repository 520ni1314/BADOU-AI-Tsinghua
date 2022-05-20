# author: orea

"""
ʹ��PCA����������X��K�׽�ά����Z
"""
 
import numpy as np
 

class CPCA(object):

    def __init__(self, X, K):
        '''
        :param X,ѵ����������X
        :param K,X�Ľ�ά����Ľ�������XҪ������ά��k��
        '''
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []
        
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()
        
    def _centralized(self):
        '''����X�����Ļ�'''
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX
        
    def _cov(self):
        '''����������X��Э�������C'''
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        return C
        
    def _U(self):
        '''��X�Ľ�άת������U, shape=(n,k), n��X������ά��������k�ǽ�ά���������ά��'''
        a,b = np.linalg.eig(self.C)
        ind = np.argsort(-1*a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U
        
    def _Z(self):
        '''����Z=XU��ά����Z, shape=(m,k), n������������k�ǽ�ά����������ά������'''
        Z = np.dot(self.X, self.U)
        return Z
        
        
if __name__=='__main__':
  import matplotlib.pyplot as plt
  from sklearn import datasets

  x, y = datasets.load_iris(return_X_y=True) #�������ݣ�x��ʾ���ݼ��е��������ݣ�y��ʾ���ݱ�ǩ
  my_pca = CPCA(x, K=2) #��ԭʼ���ݽ��н�ά��������reduced_x��
  reduced_x = my_pca.X
  red_x, red_y = [],[]
  blue_x, blue_y = [],[]
  green_x, green_y = [],[]
  for i in range(len(reduced_x)): #���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı���
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

  plt.scatter(red_x, red_y, c='r', marker='x')
  plt.scatter(blue_x, blue_y, c='b', marker='D')
  plt.scatter(green_x, green_y, c='g', marker='.')
  plt.show()
