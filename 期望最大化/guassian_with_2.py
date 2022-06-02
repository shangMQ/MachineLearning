# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
期望最大化算法在GMM（高斯混合模型）的应用
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances


# mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    style = 'sklearn'
    # style = 'hand'

    np.random.seed(0)
    # 数据集1
    # 多维分布的均值
    mu1_fact = (0, 0)
    # 创建一个3*3的标准矩阵作为协方差矩阵
    cov_fact = np.identity(2)
    data1 = np.random.multivariate_normal(mu1_fact, cov_fact, 400)

    # 数据集2
    # 多维分布的均值
    mu2_fact = (2, 2)
    # 创建一个3*3的标准矩阵作为协方差矩阵
    cov_fact = np.identity(2)
    # 生成一个多元正态分布
    data2 = np.random.multivariate_normal(mu2_fact, cov_fact, 100)

    # 合并数据集，构造一个混合数据集
    data = np.vstack((data1, data2))
    # 其中前400个样本属于第一个数据集，后100个样本属于第二个数据集
    y = np.array([True] * 400 + [False] * 100)

    if style == 'sklearn':
        # n_components设定混合高斯模型的个数
        # covariance_type设定协方差类型
        # tol 阈值
        # max_iter 最大迭代次数
        # n_init 初始化次数，用于产生最佳初始参数
        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        g.fit(data)
        print('类别概率:\t', g.weights_)
        print('混合模型的均值:\n', g.means_, '\n')
        print('混合模型的方差:\n', g.covariances_, '\n')
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
    else:
        num_iter = 100
        n, d = data.shape
        # 随机指定
        # mu1 = np.random.standard_normal(d)
        # print mu1
        # mu2 = np.random.standard_normal(d)
        # print mu2
        # 随机初始化均值，一个选择最小值，一个选择最大值
        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        # 随机初始化方差
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        # 随机初始化高斯分布权重
        pi = 0.5

        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi * norm1.pdf(data)
            tau2 = (1 - pi) * norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2)

            # M Step
            mu1 = np.dot(gamma, data) / np.sum(gamma)
            mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1 - gamma) * (data - mu2).T, data - mu2) / np.sum(1 - gamma)
            pi = np.sum(gamma) / n
            print(i, ":\t", mu1, mu2)
        print('类别概率:\t', pi)
        print('均值:\t', mu1, mu2)
        print('方差:\n', sigma1, '\n\n', sigma2, '\n')

    # 预测分类
    # 根据最终拟合的模型构建高斯模型
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    # 根据用户数据，计算相关的高斯值
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)
    # 数据可视化
    fig = plt.figure(figsize=(13, 7), facecolor='w')
    ax = fig.add_subplot(121)
    ax.scatter(data[:, 0], data[:, 1], c='b', s=30, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(u'initial data', fontsize=18)
    ax = fig.add_subplot(122)
    # 计算计算值与真实值的误差
    print(pairwise_distances([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean'))
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean')
    print(order)
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    acc = np.mean(y == c1)
    print(u'准确率：%.2f%%' % (100*acc))
    ax.scatter(data[c1, 0], data[c1, 1], c='r', s=30, marker='o')
    ax.scatter(data[c2, 0], data[c2, 1], c='g', s=30, marker='^')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(u'EM for GMM', fontsize=18)
    plt.tight_layout()
    plt.show()

    # 多元高斯概率分布概率密度函数pdf的可视化
    # 画图
    N, M = 100, 100  # 横纵各采样多少个值
    x1_min, x1_max = data[:, 0].min(), data[:, 0].max()  # 第0列的范围
    x2_min, x2_max = data[:, 1].min(), data[:, 1].max()  # 第1列的范围

    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    fx1 = np.array(norm1.pdf(x_show).reshape(-1,1)) # 预测值
    fx2 = np.array(norm2.pdf(x_show).reshape(-1,1)) # 预测值

    # pdf概率密度函数3d可视化
    fig = plt.figure(figsize=(13, 7), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_show[:,0], x_show[:,1], fx1, c='r', s=30, marker='o', depthshade=True)
    ax.scatter(x_show[:,0], x_show[:,1], fx2, c='g', s=30, marker='^', alpha=0.7, depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('pdf')
    ax.set_title(u'Probability Density Functions for Different Gaussian Distributions', fontsize=18)
    plt.show()
