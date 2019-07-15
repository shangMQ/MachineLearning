# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:00:50 2019
具有噪声的基于密度的空间聚类应用
@author: Kylin
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#1. 加载数据集
X, y = make_blobs(random_state=0, n_samples=12)

fig = plt.figure()
fig.subplots_adjust(wspace=0.8, hspace=0.4)
#2. 拟合模型，两个参数min_samples和eps取不同值时的簇分类不同
colors = ['r', 'b', 'g', 'y', 'm', 'k']
mins = [3, 4]
epses = [1., 1.5, 2.]
for i in range(2):
    for j in range(3):
        dbscan = DBSCAN(min_samples=mins[i], eps=epses[j])
        clusters = dbscan.fit_predict(X)
        print("min_sample = {}, eps = {}时的聚类结果为：{}".format(mins[i], epses[j], clusters))
        ax = fig.add_subplot(2, 3, i*3+j+1)
        for k in range(len(X)):
            ax.scatter(X[k:,0], X[k:,1], c=colors[clusters[k]], s=30)
        title = "min_sample = {}, eps = {}".format(mins[i], epses[j])
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

plt.suptitle("DBSCAN with different parameters")

