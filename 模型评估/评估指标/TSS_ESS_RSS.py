# -*- coding:utf-8 -*-
"""
TSS_ESS_RSS: TSS >= ESS + RSS
TSS: Total Sum of Squares 总方差和
ESS: Explained Sum of Squares 预测值和样本均值的方差和
RSS: Residual Sum of Sqaures 残差平方和，也就是误差平方和

R2 = 1 - RSS/TSS
:Author: Shangmengqi@tsingj.com
:Last Modified by: Shangmengqi@tsingj.com
"""
import numpy as np
import warnings
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class GradientDescent(object):
    """
    简单梯度下降法
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

    def __init__(self, eta=0.01, n_iter=200):
        """
        学习率默认0.01， 训练次数默认为50次
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
        self.w_ = np.zeros(2)
        self.cost_ = []
        self.ess_ = []
        self.rss_ = []
        n = X.shape[0]

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = output - y
            J0 = errors.sum() / n
            # n是样本数
            # 注意(X.dot(errors))计算出来的是一个(n,)维数组
            J1 = (X.dot(errors)).sum() / n
            self.w_[0] -= self.eta * J0
            self.w_[1] -= self.eta * J1
            cost = (errors ** 2).sum() / (2.0 * n)
            rss = (errors ** 2).sum()
            ess = ((self.net_input(X) - np.average(y)) ** 2).sum()
            self.cost_.append(cost)
            self.rss_.append(rss)
            self.ess_.append(ess)
        return self.w_, self.cost_

    def net_input(self, X):
        """计算所有X的净输入"""
        net = np.dot(X, self.w_[1]) + self.w_[0]
        return net


def main():
    # 相关设置
    warnings.filterwarnings(action='ignore')
    np.set_printoptions(suppress=True)
    mpl.rcParams['font.sans-serif'] = u'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False

    # 加载数据
    np.random.seed(0)
    X = np.linspace(0, 10, 100)
    y = 2 * X + np.random.randn(100)
    tss = ((y - np.average(y)) ** 2).sum()

    # 使用梯度下降预测
    gd = GradientDescent(eta=0.001)
    gd.fit(X, y)
    liner = gd.w_[1] * X + gd.w_[0]

    # 绘制图像
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    ax = fig.add_subplot(1, 3, 1)
    plt.scatter(X, y, color='red')
    plt.plot(X, liner, linewidth=3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Data and linear")

    ax = fig.add_subplot(1, 3, 2)
    iternum = np.arange(gd.n_iter)
    plt.plot(iternum, gd.cost_)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("The change of Cost function")

    ax = fig.add_subplot(1, 3, 3)
    tss = [tss] * gd.n_iter
    r_ess = np.array(gd.rss_) + np.array(gd.ess_)
    plt.plot(iternum, tss, label="TSS(Total Sum of Squares)")
    plt.plot(iternum, gd.rss_, label="RSS(Residual Sum of Squares)")
    plt.plot(iternum, gd.ess_, label="ESS(Explained Sum of Squares)")
    plt.plot(iternum, r_ess, label="RSS + ESS")
    plt.xlabel("Epochs")
    plt.ylabel("Sum of different Squares")
    plt.legend()
    plt.title("TSS = RSS + ESS")
    plt.suptitle("Linear Regression with Gradient Descent Algorithm")
    plt.show()


if __name__ == "__main__":
    main()