import numpy as np

class Perception(object):
    """
        参数：
        ------
        eta：float
             学习率(取值范围0.0-1.0)
        n_iter: int
             权重向量的训练次数
        ------
        
        属性：
        ------
        w_: array
            一维权重向量
        errors：list
            记录神经元判断错误的次数列表
        ------
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """
            拟合训练数据
            
            参数：
            ------
            X: numpy.array
               shape = [n, m]
               训练样本，n是样本数，m是每个样本的特征数
            y：numpy.array
               shape = [n_samples]
               被训练的数据集的真实类标
            ------
            
            returns：
            ------
            self：object
            ------
        """
        
        """
            将self.w_中的权值初始化为一个（m+1）维零向量。
            m是每个样本的特征数，在此基础上增加一个0权值列（也就是阈值）
        """
        self.w_ = np.zeros(1 + X.shape[1]) 
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y): #xi是每一个样本，包含m个特征，y是真实标记
                #计算预测与实际值之间的误差再乘以学习率
                update = self.eta * (target - self.predict(xi))
                #更新权重
                self.w_[1:] += update * xi
                self.w_[0] += update
                #将每轮迭代过程中分类错误的样本数量收集起来，便于后续的评判。
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """计算净输入"""
        #np.dot()计算点积
        return np.dot(X, self.w_[1:]) + self.w[0] 
    
    def predict(self, X):
        """用于计算权重更新时的类标，或者预测类标"""
        """
            np.where(condition, X, Y)用法：
                满足条件输出X，否则输出Y
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
                
        
        
        
        