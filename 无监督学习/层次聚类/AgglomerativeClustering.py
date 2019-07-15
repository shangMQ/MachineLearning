# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:45:59 2019
凝聚算法Agglomerative Clustering
注意：没有predict算法
@author: Kylin
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#加载数据
X, y = make_blobs(random_state=1)

#拟合模型
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

#绘制图像
colors = ['r', 'b', 'g']
for i in range(len(X)):
    plt.scatter(X[i:,0], X[i:,1], c=colors[assignment[i]], s=30)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Agglomerative Clustering")