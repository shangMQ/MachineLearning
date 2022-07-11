# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
根据身高体重元素绘制GMM等高线图——即边界图
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# from matplotlib.font_manager import FontProperties
# font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
# fontproperties=font_set


def expand(a, b):
    """
    左右边界各扩展5%，以便包含所有可能数据
    :param a:
    :param b:
    :return:
    """
    d = (b - a) * 0.05
    return a-d, b+d


if __name__ == '__main__':
    # 读取数据
    data = np.loadtxt('HeightWeight.csv', delimiter=',', skiprows=1)
    print(data.shape)
    # 数据集第一列为性别label，后两列为特征数据
    y, x = np.split(data, [1, ], axis=1)
    # 划分训练集和测试集
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.6, random_state=0)

    # 使用高斯混合模型
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)

    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    # 拟合模型
    gmm.fit(x)
    print('均值 = \n', gmm.means_)
    print('方差 = \n', gmm.covariances_)

    # 预测结果
    y_hat = gmm.predict(x)
    y_test_hat = gmm.predict(x_test)

    # 分析模型预测的准确性
    change = (gmm.means_[0][0] > gmm.means_[1][0])
    if change:
        z = y_hat == 0
        y_hat[z] = 1
        y_hat[~z] = 0
        z = y_test_hat == 0
        y_test_hat[z] = 1
        y_test_hat[~z] = 0
    # 计算准确率
    acc = np.mean(y_hat.ravel() == y.ravel())
    acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
    acc_str = '训练集准确率：%.2f%%' % (acc * 100)
    acc_test_str = '测试集准确率：%.2f%%' % (acc_test * 100)
    print(acc_str)
    print(acc_test_str)

    # 绘制等高线，查看混合高斯模型的边界
    # 设置等高线前景和背景色
    cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])
    # 生成等高线数据
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    print(grid_test.shape)
    # 预测数据
    grid_hat = gmm.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)

    if change:
        z = grid_hat == 0
        grid_hat[z] = 1
        grid_hat[~z] = 0
    plt.figure(figsize=(7, 6), facecolor='w')
    # plt.pcolormesh的作用在于能够直观表现出分类边界
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    # 绘制训练集预测结果，圆形
    train = plt.scatter(x[:, 0], x[:, 1], s=50, c=y.flatten(), marker='o', cmap=cm_dark, edgecolors='k')
    # 绘制测试集的分类结果，三角形
    test = plt.scatter(x_test[:, 0], x_test[:, 1], s=60, c=y_test.flatten(), marker='^', cmap=cm_dark, edgecolors='k')
    plt.legend((train, test), ('train', 'test'), loc=1)

    # 预测概率
    p = gmm.predict_proba(grid_test)
    print(f"predict prob：\n{p[:10]}...")
    # 取得第1列的概率，即属于女生的概率
    p = p[:, 0].reshape(x1.shape)
    # 绘制边界图，并添加三条轮廓线
    CS = plt.contour(x1, x2, p, levels=(0.1, 0.5, 0.8), colors=list('rgb'), linewidths=2)
    # 为轮廓线添加相应的标识
    plt.clabel(CS, fontsize=12, fmt='%.1f', inline=True)

    # 在图例上添加计算指标text
    # 获取x轴和y轴的范围
    ax1_min, ax1_max, ax2_min, ax2_max = plt.axis()
    xx = 0.95*ax1_min + 0.05*ax1_max
    yy = 0.05*ax2_min + 0.95*ax2_max
    plt.text(xx, yy, acc_str, fontsize=12)
    yy = 0.1*ax2_min + 0.9*ax2_max
    plt.text(xx, yy, acc_test_str, fontsize=12)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.xlabel('身高(cm)', fontsize=13)
    plt.ylabel('体重(kg)', fontsize=13)
    plt.title('EM算法估算GMM的参数', fontsize=15)
    plt.show()
