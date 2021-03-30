# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
    使用blob数据集拟合Kmeans聚类模型，含聚类的评价方法——轮廓系数
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from matplotlib import cm


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


def evaluateCluster(X, y):
    """
    使用轮廓系数度量聚类质量
    :param X:
    :param y:
    :return:
    """
    # 计算轮廓系数
    cluster_labels = np.unique(y)  # 获取聚类类别
    n_cluster = cluster_labels.shape[0]  # 查看类别个数
    silhouette_values = silhouette_samples(X, y, metric='euclidean')  # 以欧式距离为度量方式，计算轮廓系数

    # 轮廓系数可视化
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        # 找到类别C的样本的轮廓系数
        c_silhouette_values = silhouette_values[y == c]
        c_silhouette_values.sort()

        y_ax_upper += len(c_silhouette_values)
        color = cm.jet(i / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_values, height=1, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_values)

        # print("yticks:", yticks)
        # print("y_ax_lower:", y_ax_lower)

    # 为了便于评价，添加轮廓系数的平均值
    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels+1)
    plt.ylabel("Cluster")
    plt.xlabel("Silhouette Coefficient")
    plt.show()

    return silhouette_values


def main():
    # 1. 加载数据
    X, y = loadData()

    # 2. 数据可视化
    plotData(X, y, 'Initial')

    # 3. 使用kmeans算法
    kmean = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    kmean.fit(X) # Kmeans是无监督学习方法，只需要传递特征数据即可
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

    # 5. 聚类结果评价
    print(evaluateCluster(X, y_predict).shape)


if __name__ == "__main__":
    main()
