# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
  利用blob生成数据集，并使用kmeans进行聚类分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def loadData():
    X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
    return X, y


def plotData(X, y, message):
    plt.scatter(X[:,0], X[:,1], c=y, marker='o', s=50, edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(message)
    plt.grid()
    plt.show()


def main():
    # 1. 加载数据
    X, y = loadData()

    # 2. 数据可视化
    plotData(X, y, 'Initial')

    # 3. 使用kmeans算法
    kmean = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    kmean.fit(X, y)
    y_predict = kmean.predict(X)
    plotData(X, y_predict, 'Predict')

    # 4. 聚类对比
    # 质心存放在kmeans模型的cluster_centers_属性中
    plt.scatter(X[y_predict==0, 0], X[y_predict==0, 1], s=50, c='lightgreen', marker='s', label='cluster 1')
    plt.scatter(X[y_predict==1, 0], X[y_predict==1, 1], s=50, c='orange', marker='s', label='cluster 2')
    plt.scatter(X[y_predict==2, 0], X[y_predict==2, 1], s=50, c='lightBlue', marker='s', label='cluster 3')
    plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s=250, c='red', marker='*', label='centroids')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
