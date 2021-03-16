# /usr/bin/python
# -*- encoding:utf-8 -*-
"""
模拟Bagging
一种集成学习方式

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def f(x):
    """
    无噪声的真实值
    :param x:
    :return:
    """
    return 0.5*np.exp(-(x+3) ** 2) + np.exp(-x**2) + 0.5*np.exp(-(x-3) ** 2)


if __name__ == "__main__":
    # 1. 生成数据
    np.random.seed(0)
    N = 200
    x = np.random.rand(N) * 10 - 5  # [-5,5)
    x = np.sort(x)
    # 假设训练集的label中含有噪声
    y = f(x) + 0.05*np.random.randn(N)
    x.shape = -1, 1

    # 2. 准备模型
    degree = 6
    n_estimators = 100
    max_samples = 0.5

    # 使用L2岭回归
    ridge = RidgeCV(alphas=np.logspace(-3, 2, 20), fit_intercept=False)
    ridged = Pipeline([('poly', PolynomialFeatures(degree=degree)), ('Ridge', ridge)])

    # bagging回归
    bagging_ridged = BaggingRegressor(ridged, n_estimators=n_estimators, max_samples=max_samples)

    # 决策树回归
    dtr = DecisionTreeRegressor(max_depth=15)

    # 3. 构建bagging
    regs = [
        ('DecisionTree', dtr),
        ('Ridge(%d Degree)' % degree, ridged),
        ('Bagging Ridge(%d Degree)' % degree, bagging_ridged),
        ('Bagging DecisionTree', BaggingRegressor(dtr, n_estimators=n_estimators, max_samples=max_samples))]

    # 4. 生成测试数据
    x_test = np.linspace(1.1*x.min(), 1.1*x.max(), 1000)

    # 5. 绘图
    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6), facecolor='w')
    plt.plot(x, y, 'ro', mec='k', label='Training Data')
    plt.plot(x_test, f(x_test), color='k', lw=3, ls='-', label='Real Data')

    clrs = '#FF2020', 'm', 'y', 'g'
    for i, (name, reg) in enumerate(regs):
        reg.fit(x, y)
        label = '%s, $R^2$=%.3f' % (name, reg.score(x, y))
        y_test = reg.predict(x_test.reshape(-1, 1))
        plt.plot(x_test, y_test, color=clrs[i], lw=(i+1)*0.5, label=label, zorder=6-i)
    plt.legend(loc='upper left', fontsize=11)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Fit: samples_rate(%.1f), n_trees(%d)' % (max_samples, n_estimators), fontsize=15)
    plt.ylim((-0.2, 1.1*y.max()))
    plt.tight_layout(2)
    plt.grid(b=True, ls=':', color='#606060')
    plt.show()
