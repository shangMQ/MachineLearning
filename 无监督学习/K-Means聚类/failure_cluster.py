# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:42:30 2019
KMeans方法假设所有方向对每个簇都是同等重要的，对一些特别形状的数据无法很好的处理
@author: Kylin
"""
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

#加载数据
X, y = make_blobs(n_samples=600, random_state=170)
rng = np.random.RandomState(74)

#变换数据（对数据进行拉伸）
transformation = rng.normal(size=(2,2))
X = np.dot(X, transformation)

#拟合模型
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

#绘制聚类图形
fig = plt.figure("失败的聚类1")
colors = ['r', 'g', 'b']
#绘制聚类后的数据点
for i in range(len(X)):
    plt.scatter(X[i:,0], X[i:,1], c = colors[y_pred[i]])

#绘制聚类的质心
centers = kmeans.cluster_centers_
for j in range(3):
    plt.scatter(centers[j:,0], centers[j:,1], marker='^', c='k', s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Example 1")

fig2 = plt.figure("失败的聚类2")
#生成第二组用于模拟的数据
X2, y2 = make_moons(n_samples=200, noise=0.05, random_state=0)

#将数据集聚类成两个簇
kmeans2 = KMeans(n_clusters=2)
kmeans2.fit(X2)
y_pred2 = kmeans.predict(X2)

#绘制聚类后的数据点和质心
for i in range(len(X2)):
    plt.scatter(X2[i:,0], X2[i:,1], c = colors[y_pred2[i]])
centers2 = kmeans2.cluster_centers_
for j in range(2):
    plt.scatter(centers2[j:,0], centers2[j:,1], marker='^', c='k', s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Example 2")