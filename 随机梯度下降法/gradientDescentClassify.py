# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:12:40 2019
利用随机梯度下降实现鸢尾花分类
多元线性回归问题
@author: Kylin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class gradientdescent(object):
    """
    简单梯度下降算法
    参数：
    -----------
    eta:float类型，学习率【0,1】
    n_iter:int类型，训练次数
    -----------
    属性：
    -----------
    w_:1darray, 有两个系数，单变量线性回归的系数向量
    errors_:list, 每次训练时错误分类的个数
    -----------
    """
    def __init__(self, eta=0.01, n_iter=1000):
        """
        学习率默认0.01， 训练次数默认为1000次
        """
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """
        训练并拟合数据
        参数：
        -------------
        X：数组，shape={n_samples}
            训练向量，n_samples是样本数
        y：shape={n_samples}
            目标值
        -------------
        返回值：
        self.w_和self.cost_
        """
        n = len(X[0])
        self.w_ = np.zeros(n+1)
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = output - y
            
            J0 = errors.sum()/n
            self.w_[0] -= self.eta * J0
            
            for j in range(1,n+1):    
                J1 = (X[:,j-1].dot(errors)).sum()/n
                self.w_[j] -= self.eta * J1
            
            cost = (errors**2).sum() /(2.0*n)
            self.cost_.append(cost)
            
        return self.w_, self.cost_
    
    def net_input(self, X):
        """计算所有X的净输入"""
        #net的维度是(n,)
        net = np.dot(X, self.w_[1:]) + self.w_[0]
        return net

if __name__ == "__main__":
    dataX = [[-2,3],
             [-1,-1],
             [0,1],
             [1,2],
             [2,2],
             [2,3],
             [3,1]]
    datay = [6, -4, 4, 9, 11, 14, 10]
    X = np.array(dataX)
    y = np.array(datay)
    classify = gradientdescent()
    w_, cost_ = classify.fit(X, y)
    print(w_)
    
    
