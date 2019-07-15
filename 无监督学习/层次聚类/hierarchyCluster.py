# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:54:56 2019
使用scipy提供的树状图可视化层次聚类
@author: Kylin
"""
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#1. 加载数据
X, y = make_blobs(n_samples=12, random_state=0)
fig, axes = plt.subplots(1,2)
fig.subplots_adjust(wspace=0.2, hspace=0.4)
axes[0].scatter(X[:,0], X[:,1],s=30)
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].set_title("Original Data")

#2. 将ward聚类应用于数组X上,返回的数组表示聚合聚类时跨越的距离
linkage_array = ward(X)

#3. 绘制树状图
dendrogram(linkage_array)

#4. 在书中标记划分成三个簇的位置
axes[1] = plt.gca()
bounds = axes[1].get_xbound()
axes[1].plot(bounds, [7.25, 7.25], '--', c='k')
axes[1].plot(bounds, [4, 4], '--', c='k')

axes[1].text(bounds[1], 7.25, 'two clusters', va='center', fontdict={'size':15})
axes[1].text(bounds[1], 4, 'three clusters', va='center', fontdict={'size':15})
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Cluster Distance")
axes[1].set_title("dendrogram")

plt.suptitle("Hierachy Cluster")
