# -- coding:utf-8 --
import numpy as np
import scipy.linalg as sl
import scipy as sp


def Ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    :param data:  样本点
    :param model: 假设模型
    :param n: 生成模型所需最少的样本点
    :param k: 最大迭代次数
    :param t: 判断点满足模型的条件
    :param d: 拟合较好时，需要的样本点最少的个数
    :param debug:
    :param return_all:
    bestfit - 最优拟合解，如果未找到，返回nil
    """
    iterations=0
    bestfit=None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs,test_idxs = random_partition(n,data.shape[0])# 训练样本和测试样本
        maybe_inliers = data[maybe_idxs,:] # 训练样本的每行数据
        test_points = data[test_idxs]
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points,maybemodel)
        also_idxs = test_idxs[test_err < t] # 满足条件的索引
        also_inliers = data[also_idxs,:]
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )

        print("d=",d)
        if (len(also_inliers) > d ):
            betterdata = np.concatenate((maybe_inliers,also_inliers ))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata,bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs,also_idxs))
        iterations  += 1
        if bestfit is None:
            raise ValueError("did't meet fit acceptance criteria")
        if return_all:
            return bestfit,{"inliers":best_inlier_idxs }
        else:
            return  bestfit




def random_partition(n,n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1,idxs2


class LinearLeastSquareModel :
    def __init__(self, input_columns, output_columns, debug=False) :
        self.input_columns=input_columns
        self.output_columns=output_columns
        self.debug=debug

    def fit(self, data) :
        A=np.vstack([data[:, i] for i in self.input_columns]).T
        B=np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s=sl.lstsq(A, B)
        return x

    def get_error(self, data, model) :
        A=np.vstack([data[:, i] for i in self.input_columns]).T
        B=np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit=sp.dot(A, model)
        err_per_point=np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point

def test():
    # 生成数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples,n_inputs)) # 随机生成0-20之间的500个数据：行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs,n_outputs ))# 随机线性度，随机生成一个斜率
    B_exact = sp.dot(A_exact,perfect_fit)
    # 加入噪声
    A_noisy = A_exact + np.random.normal(size = A_exact.shape)
    B_noisy = B_exact + np.random.normal(size = B_exact.shape)

    if 1 :
        # 添加局外点
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])# 获取索引
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers ] # 100个局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers ,n_inputs ))
        B_noisy[outlier_idxs] = 50 * np.random.normal(size = (n_outliers,n_outputs))

    all_data = np.hstack((A_noisy,B_noisy))# 500行两列
    input_columns = range(n_inputs )
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    model = LinearLeastSquareModel(input_columns,output_columns,debug= debug)
    linear_fit, resids,rank,s = sp.linalg.lstsq(all_data[:,input_columns],all_data[:,output_columns])
    ransac_fit,ransac_data = Ransac(all_data,model,50,1000,7e3,300,debug=debug,return_all= True)
    if 1:
        import pylab
        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs]
        if 1 :
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else :
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()
if __name__ == '__main__':
    test()

