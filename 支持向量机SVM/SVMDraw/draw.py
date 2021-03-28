#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
探索不同的核函数下的数据预测情况
"""
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 1. 读入数据
    data = pd.read_csv('bipartition.txt', sep='\t', header=None)
    x, y = data[[0, 1]], data[2]
    print(x)
    print(y)

    # 2. 构建分类器
    clf_param = (('linear', 0.1), ('linear', 0.5), ('linear', 1), ('linear', 2), # 线性核
                ('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),  # rbf核
                ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100))

    # 3. 构建边界数据
    x1_min, x2_min = np.min(x, axis=0)
    x1_max, x2_max = np.max(x, axis=0)
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])

    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False

    # 4. 采用不同的SVM模型对数据进行拟合分析，并绘制相关决策边界图
    plt.figure(figsize=(13, 9), facecolor='w')

    for i, param in enumerate(clf_param):
        clf = svm.SVC(C=param[1], kernel=param[0])

        if param[0] == 'rbf':
            clf.gamma = param[2]
            title = 'rbf, C=%.1f, $\gamma$ = %.1f' % (param[1], param[2])
        else:
            title = 'linear, C=%.1f' % param[1]

        # 拟合数据
        clf.fit(x, y)
        y_hat = clf.predict(x)
        print('准确率：', accuracy_score(y, y_hat))

        # 查看拟合效果
        print(title)
        print('支撑向量的数目：', clf.n_support_)
        print('支撑向量的系数：', clf.dual_coef_)
        print('支撑向量：', clf.support_)

        # 画图
        plt.subplot(3, 4, i+1)
        grid_hat = clf.predict(grid_test)       # 预测分类值
        grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
        plt.scatter(x[0], x[1], c=y, edgecolors='k', s=40, cmap=cm_dark)      # 样本的显示
        plt.scatter(x.loc[clf.support_, 0], x.loc[clf.support_, 1], edgecolors='k', facecolors='none', s=100, marker='o')   # 支撑向量
        z = clf.decision_function(grid_test)
        # print('z = \n', z)
        print('clf.decision_function(x) = ', clf.decision_function(x))
        print('clf.predict(x) = ', clf.predict(x))
        z = z.reshape(x1.shape)
        print(z.shape)
        plt.contour(x1, x2, z, colors=list('kgrgk'), linestyles=['--', ':', '-', ':', '--'],
                    linewidths=1, levels=[-1, -0.5, 0, 0.5, 1])
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(title, fontsize=12)
    plt.suptitle('Different SVM Models', fontsize=16)
    plt.tight_layout(1.4)
    plt.subplots_adjust(top=0.92)
    plt.show()
