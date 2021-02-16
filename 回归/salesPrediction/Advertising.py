#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
根据相关广告数据，使用线性模型预测销量
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pprint import pprint


if __name__ == "__main__":
    show = False
    path = './Advertising.csv'

    # 1. pandas读入数据
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    # 输出特征间的相关系数
    print('皮尔森相关系数 = \n', x.corr())
    print(x)
    print(y)
    print("feature shape : ", x.shape)
    print("target shape : ", y.shape)

    # 2. 数据可视化
    mpl.rcParams['font.sans-serif'] = 'Arial Unicode MS'
    mpl.rcParams['axes.unicode_minus'] = False
    if show:
        # 绘制1, 整体特征与目标的可视化
        plt.figure(facecolor='white')
        plt.plot(data['TV'], y, 'ro', label='TV', mec='k')
        plt.plot(data['Radio'], y, 'g^', mec='k', label='Radio')
        plt.plot(data['Newspaper'], y, 'mv', mec='k', label='Newspaer')
        plt.legend(loc='lower right')
        plt.xlabel('广告花费', fontsize=16)
        plt.ylabel('销售额', fontsize=16)
        plt.title('广告花费与销售额对比数据', fontsize=18)
        plt.grid(b=True, ls=':')
        plt.show()

        # 绘制2，单个特征与目标值的可视化
        plt.figure(facecolor='w', figsize=(9, 10))
        plt.subplot(311)
        plt.plot(data['TV'], y, 'ro', mec='k')
        plt.title('TV')
        plt.grid(b=True, ls=':')
        plt.subplot(312)
        plt.plot(data['Radio'], y, 'g^', mec='k')
        plt.title('Radio')
        plt.grid(b=True, ls=':')
        plt.subplot(313)
        plt.plot(data['Newspaper'], y, 'b*', mec='k')
        plt.title('Newspaper')
        plt.grid(b=True, ls=':')
        plt.tight_layout(pad=2)
        # plt.savefig('three_graph.png')
        plt.show()

    # 3. 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # 4. 使用线性模型进行拟合
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("=====模型相关参数=======")
    print("权重：", model.coef_)
    print("截距：", model.intercept_)

    # 排序
    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    # 4. 预测
    y_test_pred = model.predict(x_test)
    # 计算均方误差
    mse = np.mean((y_test_pred - np.array(y_test)) ** 2)  # Mean Squared Error
    # 计算均方根误差
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mse_sys = mean_squared_error(y_test, y_test_pred)

    # 5. 模型评估
    print("=======模型评估=======")
    print('MSE = ', mse)
    print('MSE(System Function) = ', mse_sys)
    print('MAE = ', mean_absolute_error(y_test, y_test_pred))
    print('RMSE = ', rmse)
    print('Training R2 = ', model.score(x_train, y_train))
    print('Training R2(System) = ', r2_score(y_train, model.predict(x_train)))
    print('Test R2 = ', model.score(x_test, y_test))

    error = y_test - y_test_pred
    np.set_printoptions(suppress=True)
    print('error = ', error)
    plt.hist(error, bins=20, color='g', alpha=0.6, edgecolor='k')
    plt.xlabel("Error")
    plt.ylabel("Num")
    plt.title("Histogram for Error")
    plt.show()

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='真实数据')
    plt.plot(t, y_test_pred, 'g-', linewidth=2, label='预测数据')
    plt.legend(loc='upper left')
    plt.title('线性回归预测销量', fontsize=18)
    plt.grid(b=True, ls=':')
    plt.show()
