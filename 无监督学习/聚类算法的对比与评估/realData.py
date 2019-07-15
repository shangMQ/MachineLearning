# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:49:47 2019
用真实值评估聚类
使用ARI和轮廓系数来比较K均值、凝聚聚类和DBSCAN算法
@author: Kylin
"""
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
import matplotlib.pyplot as plt

#1. 加载数据集
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

#2. 标准化数据
scaler = StandardScaler()
scaler.fit(X)
scaled = scaler.transform(X)

#3. 创建一个随机的簇分配，作为参考
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
rc_ARI = adjusted_rand_score(y, random_clusters)
rc_ss = silhouette_score(scaled,random_clusters)
#4. 绘制分簇结果
fig = plt.figure(figsize=(18,7))
fig.subplots_adjust(wspace=0.5)
colors = ['r', 'b', 'g', 'y', 'k']
#首先绘制随机分簇的结果
ax0 = fig.add_subplot(1,4,1)
for i in range(len(X)):
    ax0.scatter(scaled[i:,0], scaled[i:,1],c=colors[random_clusters[i]], s=30)
ax0.set_xlabel("Feature 1")
ax0.set_ylabel("Feature 2")
title0 = "Random Assignment: ARI={:.2f}, ss={:.2f}".format(rc_ARI, rc_ss)
ax0.set_title(title0)

#绘制kmeans算法的分簇结果
kmeans = KMeans(n_clusters=2)
kmeans_clusters = kmeans.fit_predict(scaled)
kmeans_ARI = adjusted_rand_score(y, kmeans_clusters)
kmeans_SS = silhouette_score(scaled, kmeans_clusters)
ax1 = fig.add_subplot(1,4,2)
for i in range(len(X)):
    ax1.scatter(scaled[i:,0], scaled[i:,1],c=colors[kmeans_clusters[i]], s=30)
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
title1 = "KMeans: ARI={:.2f}, ss={:.2f}".format(kmeans_ARI, kmeans_SS)
ax1.set_title(title1)

#绘制聚合聚类算法的分簇结果
ac = AgglomerativeClustering(n_clusters=2)
ac_clusters = ac.fit_predict(scaled)
ac_ARI = adjusted_rand_score(y, ac_clusters)
ac_ss = silhouette_score(scaled, ac_clusters)
ax2 = fig.add_subplot(1,4,3)
for i in range(len(X)):
    ax2.scatter(scaled[i:,0], scaled[i:,1],c=colors[ac_clusters[i]], s=30)
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
title2 = "AC: ARI={:.2f}, ss={:.2f}".format(ac_ARI, ac_ss)
ax2.set_title(title2)

#绘制DBSCAN聚类算法的分簇结果
dbscan = DBSCAN()
dbscan_clusters = dbscan.fit_predict(scaled)
dbscan_ARI = adjusted_rand_score(y, dbscan_clusters)
dbscan_ss = silhouette_score(scaled, dbscan_clusters)
ax3 = fig.add_subplot(1,4,4)
for i in range(len(X)):
    ax3.scatter(scaled[i:,0], scaled[i:,1],c=colors[dbscan_clusters[i]], s=30)
ax3.set_xlabel("Feature 1")
ax3.set_ylabel("Feature 2")
title3 = "DBSCAN: ARI={:.2f}, ss={:.2f}".format(dbscan_ARI, dbscan_ss)
ax3.set_title(title3)