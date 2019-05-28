# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:13:07 2019
利用计算机内存价格数据集实现回归树
@author: Kylin
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib as mpl


#设置图像的中文字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#获取数据
ram_prices = pd.read_csv("ram_price.csv")

#数据可视化
fig = plt.figure("计算机内存价格走势")
plt.semilogy(ram_prices.date, ram_prices.price)#y轴是指数型单位
plt.xlabel("year")
plt.ylabel("Price in $/Mbyte") 

#利用历史数据来预测2000年后的价格
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

#基于日期来预测价格
X_train = data_train.date[:, np.newaxis]
#利用对谁变换得到数据和目标之间更简单的关系
y_train = np.log(data_train.price)

#对比回归树和线性回归的效果
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_rag = LinearRegression().fit(X_train, y_train)

#对所有数据进行预测
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_linear = linear_rag.predict(X_all)

#对数变换逆运算
price_tree = np.exp(pred_tree)
price_linear = np.exp(pred_linear)

#图像对比
fig2 = plt.figure("回归树与线性回归对比")
plt.semilogy(data_train.date, data_train.price, label="Training Data")
plt.semilogy(data_test.date, data_test.price, label="Test Data")
plt.semilogy(ram_prices.date, price_tree, label="Tree Prediction")
plt.semilogy(ram_prices.date, price_linear, label="Linear Prediction")
plt.title("回归树与线性回归在预测计算机内存价格的对比")
plt.xlabel("year")
plt.ylabel("price")
plt.legend()
plt.show()