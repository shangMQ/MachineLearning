# -*- coding:utf-8 -*-
"""
LogisticRegression_plain逻辑回归的明文

:Author: Shangmengqi@tsingj.com
:Last Modified by: Shangmengqi@tsingj.com
"""
import numpy as np


class LogisticRegression_plain:
    """
    明文版本的逻辑回归的实现
    """
    def __init__(self, q=0.01, num_iter=10, random_state=96, shrink_step=True, batch_size=-1):
        """
        相关参数的初始化
        ----------------
        Args:
            q: double,
               学习率
            num_iter: int,
                      迭代次数
            random_state: int,
                          随机整数，用于初始化模型参数
            shrink_step: boolean,
                         学习率是否随训练轮数下降
            batch_size: double,
                        若该值小于0(如可设置为-1)，表示每次训练使用全部训练集样本
        """
        self.q = q
        self.num_iter = num_iter
        # 权重参数刚开始是空的
        self.theta = np.zeros([])
        self.random_state = random_state

        self.result = []
        self.shrink_step = shrink_step
        self.batch_size = batch_size

    def __sigmoid(self, x):
        """
        利用公式1/(1+np.exp(-x))计算关于x的逻辑回归值
        ------------------
        Args:
            x: double,
               用于计算逻辑回归值的参数
        ------------------
        Returns:
            double, x对应的逻辑回归值
        """
        return 1 / (1 + np.exp(-x))

    def __add_intercept(self, x):
        """
        添加截距
        --------
        Args:
            x: 数据
        --------
        Returns:
            ndarray 添加截距后的数据
        """
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis=1)

    def fit(self, x, y):
        """
        利用小批量梯度下降算法实现拟合函数
        -------
        Args:
            x: ndarray
               特征数据集
            y: ndarray
               标签数据集
        """
        # 1. 添加截距
        x = self.__add_intercept(x)

        # 2. 处理批大小小于0的情况
        if self.batch_size < 0:
            self.batch_size = len(x)

        np.random.seed(self.random_state)
        # 将权重设置为随机值
        self.theta = np.random.rand(x.shape[1])

        for i in range(self.num_iter):
            for j in range(0, len(x), self.batch_size):
                temp_x = x[j: j + self.batch_size]
                temp_y = y[j: j + self.batch_size]
                z = np.dot(temp_x, self.theta)

                # 计算预测值
                predict_value = self.__sigmoid(z)

                # 梯度值处理
                # 代价函数J = (label - predict)**2 /2*len(x)
                # 梯度 = J关于权重的导数 = x *（label - predict） / len(x)
                gradient = np.dot(temp_x.T, (predict_value - temp_y)) / len(temp_y)

                # 更新权重
                if self.shrink_step:
                    # 判断是否需要降低学习率
                    self.theta -= self.q * gradient / (i + 1)
                else:
                    self.theta -= self.q * gradient

    def predict(self, x):
        """
        预测函数
        实现sigmoid函数结果到类别的映射
        ---------
        Args:
            x: ndarray
               特征数据
        ---------
        Returns:
            int: 预测类别
        """
        return (self.predict_proba(x) > 0.5).astype("int")

    def predict_proba(self, x):
        """
        计算样本特征的概率
        -------------
        Args:
            x: ndarray
               特征数据
        ------------
        Returns:
            double: 概率值
        """
        x = self.__add_intercept(x)
        return self.__sigmoid(np.dot(x, self.theta))



