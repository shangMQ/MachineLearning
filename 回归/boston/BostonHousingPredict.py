#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
波士顿房价预测
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import sklearn.datasets
from pprint import pprint
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings


def not_empty(s):
    """
    判断元素是否为空
    :param s:
    :return:
    """
    return s != ''


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    np.set_printoptions(suppress=True)
    mpl.rcParams['font.sans-serif'] = u'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False

    # 1. 读取数据
    file_data = pd.read_csv('housing.data', header=None)
    data = np.empty((len(file_data), 14))

    for i, d in enumerate(file_data.values):
        # filter()函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
        # 接收两个参数，第一个为函数，第二个为序列
        # 序列的每个元素作为参数传递给函数进行判断
        # 然后返回True或False，最后将返回True的元素放到新列表中

        d = list(map(float, list(filter(not_empty, d[0].split(' ')))))
        data[i] = d

    # 数据集的特征与标签划分
    x, y = np.split(data, (13, ), axis=1)
    # 其实，datasets中也提供了波士顿数据集
    # data = sklearn.datasets.load_boston()
    # x = np.array(data.data)
    # y = np.array(data.target)
    print('样本个数：%d, 特征个数：%d' % x.shape)
    print(y.shape)
    y = y.ravel()

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

    # 3. 构建模型
    model = Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=True)),
        ('linear', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
                                fit_intercept=False, max_iter=1e3, cv=3))
    ])
    # model = RandomForestRegressor(n_estimators=50, criterion='mse')
    print('开始建模...')
    model.fit(x_train, y_train)
    linear = model.get_params('linear')['linear']
    # print(u'超参数：', linear.alpha_)
    # print(u'L1 ratio：', linear.l1_ratio_)
    # print(u'系数：', linear.coef_.ravel())

    # 4. 为了便于查看，进行排序处理
    order = y_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order, :]

    # 5. 查看预测结果
    y_pred = model.predict(x_test)

    # 6. 模型评估
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('R2:', r2)
    print('均方误差：', mse)

    # 7. 预测可视化
    t = np.arange(len(y_pred))
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', lw=2, label='Real')
    plt.plot(t, y_pred, 'g-', lw=2, label='Estimate')
    plt.legend(loc='best')
    plt.title('Prediction of Boston Housing Price', fontsize=18)
    plt.xlabel('No. of sample', fontsize=15)
    plt.ylabel('Housing price', fontsize=15)
    plt.grid()
    plt.show()
