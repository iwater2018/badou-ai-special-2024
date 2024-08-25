import numpy as np


class CPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, X, K):
        '''
        :param X,样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        print('样本矩阵：\n',self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本的均值：\n',mean)
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:\n',centrX)
        return centrX
    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T,self.centrX)/(ns - 1)
        print('样本矩阵X的协方差矩阵C：\n',C)
        return C
    def _U(self):
        a,b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值：\n',a)
        print('样本集的协方差矩阵C的特征向量：\n',b)
        ind = np.argsort(-1 * a) # 对a特征值进行降序排列
        # 构建K阶降维的降维转换矩阵
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U：\n'%self.K,U)
        return U
    def _Z(self):
        # Z=XU求降维矩阵Z
        Z = np.dot(self.X,self.U)
        print('X shape:',np.shape(self.X))
        print('U shape:',np.shape(self.U))
        print('Z shape:',np.shape(Z))
        print('样本降维矩阵X的降维矩阵Z:\n',Z)
        return Z

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)