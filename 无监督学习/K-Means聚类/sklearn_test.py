# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:33:19 2019
利用sklearn包中的kMeans算法
@author: Kylin
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#1. 生成二维数据集
X, y = make_blobs(random_state=1)
y.astype("int")

#2. 数据可视化
colors = ['r', 'b', 'y']
fig, axes = plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5)
axes[0].scatter(X[:,0], X[:,1], s=30)
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].set_title("Original Data")


#3. 构建聚类模型
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clustered = kmeans.labels_
print("训练集分簇结果：", clustered)

#4. 绘制分簇后的数据
for i in range(len(X)):
    axes[1].scatter(X[i:,0], X[i:,1], c = colors[clustered[i]], s=30)
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].set_title("Clustering Data")
#绘制质心
centers = kmeans.cluster_centers_
for j in range(3):
    axes[1].scatter(centers[j:, 0], centers[j:, 1], marker='^', c = "k", s=50)