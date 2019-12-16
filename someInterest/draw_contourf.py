# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:29:33 2019
绘制决策边界
数据集：blob数据集
方法：kmeans方法
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def loaddata():
    """
    加载数据集
    """
    #设定每个聚类的质心数据，还有每个blob数据的标准差
    blob_centers = np.array(
        [[ 0.2,  2.3],
         [-1.5 ,  2.3],
         [-2.8,  1.8],
         [-2.8,  2.8],
         [-2.8,  1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    
    #利用make_blob生成数据集
    X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)
    
    return X, y


def plot_data(X):
    """绘制数据集散点图"""
    plt.scatter(X[:,0], X[:,1], color="k", s=3)
    plt.xlabel(r"$X_{1}$")
    plt.ylabel(r"$X_{2}$")

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    """绘制质心的函数
       参数：
          centroids:质心元素
          weights:权重
          circle_color:质心颜色
          cross_color:边框、字体颜色
    """
    
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    
    #绘制质心散点图
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    """
    决策边界绘制函数
        参数：
            clusterer：聚类方法
            X：原始数据集中的特征向量
            resolution：生成的元素个数
            show_centroids：绘制质心
            show_xlabels/ylabels：是否显示横轴/纵轴的坐标轴
    """
    plt.figure("decision boundaries")
    
    #axis = 0时，按行找到最小元素
    mins = X.min(axis = 0) - 0.1
    maxs = X.max(axis = 0) + 0.1
    
    #meshgrid生成网格数据：xx，yy都是1000行1000列
    #xx是第一个特征的向量按行复制的，yy是第二个特征向量，按列复制
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    
    #Z得到的是将xx，yy合并后的预测值
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z需要reshape一下，reshape成和xx一样的shape
    Z = Z.reshape(xx.shape)

    #使用plt.contourf()绘制等高线图
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    
    plot_data(X)
    
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    
    plt.title("Clustering Decision Boundary")
    plt.show()

if __name__ == "__main__":
    #1. 加载数据集
    X, y = loaddata()
    
    #2. 查看数据集图像
    plot_data(X)
    
    #3. 利用KMeans方法实现聚类
    k = 5 #设定聚簇个数
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    
    #4. 绘制决策边界
    plot_decision_boundaries(kmeans, X)
    