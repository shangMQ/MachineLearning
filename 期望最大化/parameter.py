# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
GMM的调参，主要对于covariances_type
- 'full': each component has its own general covariance matrix.
          每个高斯分布都有自己的一般协方差矩阵。
- 'tied': all components share the same general covariance matrix.
          所有component的协方差矩阵相同
- 'diag': each component has its own diagonal covariance matrix.
          每个高斯分布都有自己的对角协方差矩阵
- 'spherical': each component has its own single variance.
               每个component都有自己的单一方差
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def expand(a, b, rate=0.05):
    """
    边界扩展
    :param a:
    :param b:
    :param rate:
    :return:
    """
    d = (b - a) * rate
    return a-d, b+d


def accuracy_rate(y1, y2):
    """
    计算准确率
    :param y1:
    :param y2:
    :return:
    """
    acc = np.mean(y1 == y2)
    return acc if acc > 0.5 else 1-acc


if __name__ == '__main__':
    # 生成数据
    np.random.seed(0)
    # 对角阵
    cov = np.diag((1, 2))
    # 第一个高斯分布500条数据，第二个高斯分布300条数据
    N1 = 500
    N2 = 300
    N = N1 + N2
    x1 = np.random.multivariate_normal(mean=(1, 2), cov=cov, size=N1)
    m = np.array(((1, 1), (1, 3)))
    x1 = x1.dot(m)
    x2 = np.random.multivariate_normal(mean=(-1, 10), cov=cov, size=N2)
    x = np.vstack((x1, x2))
    y = np.array([0]*N1 + [1]*N2)

    # 四种协方差类型
    types = ('spherical', 'diag', 'tied', 'full')

    # 用于记录错误率
    err = np.empty(len(types))
    # 用于记录bic
    bic = np.empty(len(types))

    for i, type in enumerate(types):
        gmm = GaussianMixture(n_components=2, covariance_type=type, random_state=0)
        gmm.fit(x)
        err[i] = 1 - accuracy_rate(gmm.predict(x), y)
        bic[i] = gmm.bic(x)

    print('错误率：', err.ravel())
    print('BIC：', bic.ravel())

    # 生成4个x轴上的位置
    xpos = np.arange(4)
    plt.figure(facecolor='w')
    ax = plt.axes()
    # 绘制条形图
    # 第一个条形图靠左0.3个单位，绘制错误率
    b1 = ax.bar(xpos - 0.3, err, width=0.3, color='#77E0A0', edgecolor='k')
    # 利用ax.twinx()绘制第二个轴，用于表示bic
    b2 = ax.twinx().bar(xpos, bic, width=0.3, color='#FF8080', edgecolor='k')
    bic_min, bic_max = expand(bic.min(), bic.max())
    plt.ylim((bic_min, bic_max))
    plt.xticks(xpos, types)
    plt.legend([b1[0], b2[0]], ('错误率', 'BIC'))
    plt.title('不同方差类型的误差率和BIC', fontsize=15)
    plt.show()

    # 利用bic最低的参数绘制边界图
    optimal = bic.argmin()
    gmm = GaussianMixture(n_components=2, covariance_type=types[optimal], random_state=0)
    gmm.fit(x)
    print('均值 = \n', gmm.means_)
    print('方差 = \n', gmm.covariances_)
    y_hat = gmm.predict(x)

    # 前景色和背景色
    cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])
    # 得到mashgrid数据
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    # 合成数据，用于绘制边界图
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = gmm.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)

    # 考虑到label的顺序可能不一致
    if gmm.means_[0][0] > gmm.means_[1][0]:
        z = grid_hat == 0
        grid_hat[z] = 1
        grid_hat[~z] = 0

    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    # 绘制真实标记
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='o', cmap=cm_dark, edgecolors='k')

    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.title('GMM调参：covariance_type=%s' % types[optimal], fontsize=15)
    plt.grid()
    plt.show()