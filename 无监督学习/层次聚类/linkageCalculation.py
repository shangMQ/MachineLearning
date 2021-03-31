# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
基于距离矩阵的层次聚类
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


def loadData():
    np.random.seed(123)
    variables = ['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
    X = np.random.random_sample([5, 3]) * 10
    df = pd.DataFrame(X, columns=variables, index=labels)
    return df, labels


def main():
    # 1. 加载数据
    data, labels = loadData()
    print("======Data=======\n", data)

    # 2. 计算距离矩阵(样本与其他样本的距离)
    row_dist = pd.DataFrame(squareform(pdist(data, metric='euclidean')), columns=labels, index=labels)
    print("========samples distance=======\n", row_dist)

    # 3. 计算关联矩阵（全连接方式：通过比较找到分布于两个簇中最不相似的样本，而完成簇的合并）
    # 每一行代表一次簇的合并
    # 矩阵的第一列和第二列分别表示每个簇中最不相似的样本
    # 第三列为这些样本的间的距离
    # 每一个簇中的样本的数量
    row_clusters = linkage(data.values, method='complete', metric='euclidean')
    rowDF = pd.DataFrame(row_clusters, columns=['row label 1', 'row label 2', 'distance', 'no. of items in cluster'],
                         index=['cluster %d' % (i+1) for i in range(row_clusters.shape[0])])
    print("=========关联矩阵========\n", rowDF)

    # 4. 聚类结果可视化
    row_dendr = dendrogram(row_clusters, labels=labels)
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.show()

    # 5. 热力图显示
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    # 是一个python字典，通过leaves键访问得到
    row_dendr = dendrogram(row_clusters, orientation='right')
    df_rowclust = data.ix[row_dendr['leaves'][::-1]]
    axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])

    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

    # 删除刻度
    ax.set_xticks([])
    ax.set_yticks([])

    for i in ax.spines.values():
        i.set_visible(False)

    fig.colorbar(cax)
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))
    plt.show()


if __name__ == "__main__":
    main()