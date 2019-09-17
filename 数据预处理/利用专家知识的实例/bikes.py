# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:21:18 2019

@author: Kylin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

def load_citibike():
    data_mine = pd.read_csv("citibike.csv")

    data_mine['one'] = 1
    
    #将starttime从object类型转换为datetime类型
    data_mine['starttime'] = pd.to_datetime(data_mine.starttime)
    
    #将starttime这一列元素作为index
    data_starttime = data_mine.set_index("starttime")
    
    #查看数据是否存在缺失值
#    print(data_mine.count())
    
    #pd.resample()对常规时间序列数据重新采样和频率转换的便捷的方法
    data_resampled = data_starttime.resample("3h").sum().fillna(0)
    
    return data_resampled.one

def drawMonth(data):
    plt.figure(figsize=(30,8))
    xticks = pd.date_range(start=data.index.min(), end=data.index.max(), freq = "D")
    plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=75, ha="left")
    plt.plot(data, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.title("Month lent numbers Fig")
    return xticks

def eval_on_features(X_train, X_test, y_train, y_test, regressor):
    """
    利用给定的机器学习模型拟合数据集，并评估其结果
    """
    pictitle = str(type(regressor)).split("'")[1].split(".")[-1] + "elavator Fig"
    regressor.fit(X_train, y_train)
    print("Test Set R2:{:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    fig2 = plt.figure(figsize=(10,5))
    plt.xticks(range(0, len(X_train)+len(X_test),8), xticks.strftime("%a %m-%d"), rotation=75, ha="left")
    plt.plot(range(n_train), y_train, color="b", label="train")
    plt.plot(range(n_train, len(y_test)+n_train), y_test, color="green", label="test")
    plt.plot(range(n_train), y_pred_train, "--", color="red", label="predict train")
    plt.plot(range(n_train, len(y_test)+n_train), y_pred, "--", color="yellow", label="predict test")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.title(pictitle)
    

if __name__ == "__main__":
    #1. 获取数据集
    data = load_citibike()
    print("查看数据前五行:\n", data.head())
    
    #2. 可视化整个月的租车情况
    xticks = drawMonth(data)#保存一下时间刻度
    
    #3.提取目标值和特征值
    y = data.values
    X = data.index.astype("int64").values.reshape(-1, 1)
    print("X.shape:", X.shape)
    
    #4. 划分训练集与测试集
    n_train = 184#前184个数据用于训练，后面的用于测试
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    #5. 选择时间作为特征
#    print("------利用随机森林模型------")
#    randforest = RandomForestRegressor(n_estimators=100, random_state=0)
#    eval_on_features(X_train, X_test, y_train, y_test, randforest)
    
#    #6. 选择hour作为特征数据
#    X = data.index.hour.values.reshape(-1,1)
#    X_train, X_test = X[:n_train], X[n_train:]
#    print("------利用随机森林模型------")
#    randforest = RandomForestRegressor(n_estimators=100, random_state=0)
#    eval_on_features(X_train, X_test, y_train, y_test, randforest)
    
    #7. 选择hour作为特征数据
    X = np.hstack([data.index.hour.values.reshape(-1,1), data.index.dayofweek.values.reshape(-1,1)])
    X_train, X_test = X[:n_train], X[n_train:]
#    print("------利用随机森林模型------")
#    randforest = RandomForestRegressor(n_estimators=100, random_state=0)
#    eval_on_features(X_train, X_test, y_train, y_test, randforest)
    
    #8. 使用简单的线性模型
    print("------使用线性回归模型------")
    lr = LinearRegression()
    eval_on_features(X_train, X_test, y_train, y_test, lr)
    
    #9. 将整数解释为分类变量
    enc = OneHotEncoder()
    X_hour_week_one_hot = enc.fit_transform(X).toarray()
    X_train, X_test = X_hour_week_one_hot[:n_train], X_hour_week_one_hot[n_train:]
    print("------利用Ridge模型------")
    ridge = Ridge()
    eval_on_features(X_train, X_test, y_train, y_test, ridge)
    
    #10. 使用hour和星期几的交互特征
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly_transformer.fit_transform(X_hour_week_one_hot)
    X_train, X_test = X_poly[:n_train], X_poly[n_train:]
    print("------利用Ridge模型------")
    ridge = Ridge()
    eval_on_features(X_train, X_test, y_train, y_test, ridge)