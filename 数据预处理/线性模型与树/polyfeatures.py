# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:18:34 2019
多项式特征应用在波士顿房价预测问题上
@author: Kylin
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

#1. 加载数据
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

#2. 缩放数据
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#3. 提取多项式特征和特征交互
poly = PolynomialFeatures(degree=2)
poly.fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("X_train shape:", X_train.shape)
print("X_train_poly shape:", X_train_poly.shape)
print("Polynomial feature names:", poly.get_feature_names())

#4. 将岭回归模型应用在不同数据集上（有效果）
print("--------Ridge--------")
ridge1 = Ridge()
ridge1.fit(X_train_scaled, y_train)
print("Initial data test Score:", ridge1.score(X_test_scaled, y_test))

ridge2 = Ridge()
ridge2.fit(X_train_poly, y_train)
print("Polynomial data test Score:", ridge2.score(X_test_poly, y_test))


#5. 利用随机森林（可能会降低其性能）
print("-----RandomForestRegressor--------")
randomforest = RandomForestRegressor(n_estimators=100)
randomforest.fit(X_train_scaled, y_train)
print("Initial data test Score:", randomforest.score(X_test_scaled, y_test))

randomforest.fit(X_train_poly, y_train)
print("Polynomial data test Score:", randomforest.score(X_test_poly, y_test))

