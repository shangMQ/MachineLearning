# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:43:08 2019
利用岭回归模型和锁套模型实现销售问题的回归
@author: Kylin
"""
#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    #1. pandas读入数据
    data = pd.read_csv('Advertising.csv')    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    print(x)
    print(y)

    #2. 将数据集分为训练集和测试集，默认将75%的数据划分到训练集中。
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # print x_train, y_train
    model1 = Ridge()
    model2 = Lasso()
    
    #3. 设置α,并拟合模型
    alpha_can = np.logspace(-3, 2, 10)
    ridge_model = GridSearchCV(model1, param_grid={'alpha': alpha_can}, cv=5)
    ridge_model.fit(x, y)
    lasso_model = GridSearchCV(model2, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x, y)
    print('Ridge回归模型的验证参数：\n', ridge_model.best_params_)
    print('Lasso回归模型的验证参数：\n', lasso_model.best_params_)

    #4. 查看预测结果的mse和rmse
    y_hat1 = ridge_model.predict(np.array(x_test))
    mse1 = np.average((y_hat1 - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse1 = np.sqrt(mse1)  # Root Mean Squared Error
    y_hat2 = lasso_model.predict(np.array(x_test))
    mse2 = np.average((y_hat2 - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse2 = np.sqrt(mse2)  # Root Mean Squared Error
    print("Ridge：mse = {:.2f}, rmse = {:.2f}".format(mse1, rmse1))
    print("Lasso：mse = {:.2f}, rmse = {:.2f}".format(mse2, rmse2))

    #5. 查看拟合效果
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat1, 'g--', linewidth=3, label='Rideg Predict')
    plt.plot(t, y_hat2, 'b-', linewidth=1, label='Lasso Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

