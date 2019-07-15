# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:39:21 2019
eps参数可以隐式地控制寻找簇的个数
@author: Kylin
"""
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#1. 加载数据集
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

#2. 将数据标准化
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#3. 拟合数据
dbscan = DBSCAN()#默认情况下eps=0.5
clusters = dbscan.fit_predict(X_scaled)

#4. 绘制分簇结果
colors = ['r', 'b', 'g', 'y', 'k']
for i in range(len(X)):    
    plt.scatter(X[i:,0], X[i:,1], c = colors[clusters[i]], s=30)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Eps=0.5")
