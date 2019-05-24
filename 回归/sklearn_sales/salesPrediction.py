# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:47:47 2019
不同营销手段对销量的影响
@author: Kylin
"""
#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    #设置字体
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    
    # 1. pandas读入数据
    path = 'Advertising.csv'
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x1 = data[['TV', 'Radio', 'Newspaper']] #x1考虑三种特征
    x2 = data[['TV', 'Radio']] #x2忽略了报纸的特征
    y = data['Sales'] #目标值销量
    print(x1)
    print(y)

    # # 绘制1
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    plt.xlabel("cost(million$)")
    plt.ylabel("sales(million$)")
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    # #
    
    #  绘制2(分别显示不同营销手段对销量的影响)
    plt.figure(figsize=(9,12))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()#进行排版
    plt.show()
    
    print("*"*5, "利用TV、radio和newspaer三个特征构建模型", "*"*5)
    x1_train, x1_test, y_train, y_test = train_test_split(x1, y, random_state=1)
    # print x_train, y_train
    linreg = LinearRegression()
    model = linreg.fit(x1_train, y_train)
    print(model)
    print("模型系数:", linreg.coef_)
    print("模型截距:", linreg.intercept_)

    
    y_hat = linreg.predict(np.array(x1_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print("(均方误差={:.2f}, 均方根误差={:.2f})".format(mse, rmse))
    #model.score()函数实际计算的就是R2决定系数
    print("R2_train =", model.score(x_train, y_train))
    print("R2_test =", model.score(x_test, y_test))
    

    t = np.arange(len(x1_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.xlabel("cost(million$)")
    plt.ylabel("sales(million$)")
    plt.title("利用TV、Radio和Newspaer拟合线性模型")
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    
    print("*"*5, "利用TV和radio两个特征构建模型", "*"*5)
    x2_train, x2_test, y_train, y_test = train_test_split(x2, y, random_state=1)
    # print x_train, y_train
    linreg = LinearRegression()
    model2 = linreg.fit(x2_train, y_train)
    print(model2)
    print("模型系数:", linreg.coef_)
    print("模型截距:", linreg.intercept_)

    #因为随机打乱了，对测试集进行排序，方便图像显示效果
    order = y_test.argsort(axis=0) #y_test是个列向量
    y_test = y_test.values[order]
    x_test = x_test.values[order,:]
    y_hat = linreg.predict(np.array(x2_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print("(均方误差={:.2f}, 均方根误差={:.2f})".format(mse, rmse))

    t = np.arange(len(x2_test))
    fig = plt.figure("利用回归进行预测")
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.xlabel("cost(million$)")
    plt.ylabel("sales(million$)")
    plt.title("利用TV和Radio拟合线性模型")
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    
