# -*- coding:utf-8 -*-
"""
gd_vs_lr
利用梯度下降法计算和逻辑回归计算的结果对比
:Author: Shangmengqi@tsingj.com
:Last Modified by: Shangmengqi@tsingj.com
"""
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)
        

def main():
    # 1. 加载数据
    boston = load_boston()

    # 2. 划分数据集
    X, y = boston.data[:, 5].reshape(-1, 1), boston.target

    # 3. 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 调用LRGD计算系数
    lr_gd = LinearRegressionGD()
    lr_gd.fit(X_scaled, y)
    lr_gd_predict = lr_gd.predict(X_scaled)
    print("=====Linear Regression Gradient Descent======")
    print(lr_gd.w_)

    # 5. 可视化损失函数
    n_iter = list(range(1, 21))
    plt.plot(n_iter, lr_gd.cost_, 'ro-')
    plt.xlabel("Iter Num")
    plt.ylabel("Loss")
    plt.xlim(0, 21)
    plt.title("Loss change")
    plt.show()

    # 6. 利用线性回归计算w
    lr = LinearRegression()
    lr.fit(X_scaled, y)
    lr_predict = lr.predict(X_scaled)
    print("=====Linear Regression======")
    print(lr.intercept_, lr.coef_)

    # 7. 对比预测结果
    plt.scatter(X_scaled, y, color='k', s=10)
    plt.scatter(X_scaled, lr_gd_predict, color='r', edgecolors='k', s=40)
    plt.scatter(X_scaled, lr_predict, color='g', edgecolors='k', s=20, alpha=0.2)
    plt.show()


if __name__ == "__main__":
    main()


