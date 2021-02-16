#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
带正则化的线性回归模型
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    # 1. pandas读入数据
    data = pd.read_csv('Advertising.csv')    # TV、Radio、Newspaper、Sales
    print(data.head())
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    print("Feature :\n", x)
    print("Target :\n", y)

    # 2. 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

    # 3. 使用模型
    lasso = Lasso()
    ridge = Ridge()
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    print('alpha_can = ', alpha_can)
    ridge_model = GridSearchCV(ridge, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model = GridSearchCV(lasso, param_grid={'alpha': alpha_can}, cv=5)
    ridge_model.fit(x_train, y_train)
    lasso_model.fit(x_train, y_train)
    print('Lasso模型超参数：\n', lasso_model.best_params_)
    print('Ridge模型超参数：\n', lasso_model.best_params_)

    # 4. 预测
    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat1 = lasso_model.predict(x_test)
    y_hat2 = ridge_model.predict(x_test)
    print("Lasso模型的R2 = ", lasso_model.score(x_test, y_test))
    print("Ridge模型的R2 = ", ridge_model.score(x_test, y_test))
    mse1 = np.average((y_hat1 - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse1 = np.sqrt(mse1)  # Root Mean Squared Error
    mse2 = np.average((y_hat2 - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse2 = np.sqrt(mse2)  # Root Mean Squared Error
    print("Lasso模型, mse = {}, rmse = {}".format(mse1, rmse1))
    print("Ridge模型, mse = {}, rmse = {}".format(mse2, rmse2))

    # 5. 可视化对比
    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = 'Arial Unicode MS'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label='真实数据')
    plt.plot(t, y_hat1, 'g-', linewidth=2, label='预测数据Lasso', alpha=0.6)
    plt.plot(t, y_hat2, 'b--', linewidth=2, label='预测数据Ridge', alpha=0.6)
    plt.title('线性回归预测销量', fontsize=18)
    plt.xlabel("Recording")
    plt.ylabel("Sales")
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.show()
