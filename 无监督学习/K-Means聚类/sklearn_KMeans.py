# !/usr/bin/python
#应用sklearn.cluster包中实现的KMeans算法
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans


def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


if __name__ == "__main__":
    N = 400 #设置样本数
    centers = 4 #设置类别数
    #make_blobs()为聚类产生数据集（数据集，标签）
    #n_features设定特征维度（默认是2），centers产生数据的中心点（默认是3）
    #cluster_std设定数据集的标准差，默认是1.0
    #data是原始数据
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    #data2生成标准差不同的数据
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1,2.5,0.5,2), random_state=2)
    
    #data3生成不平衡数据,其中第一类的全都有，第二类的有50个，第三类的有20个，第四类的有5个
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)#生成相应的标签值
    
    """n_clusters生成质心数（产生簇数）
       init设置初始质心的选择方法:
           "random" 随机从训练数据中选取初始质心
           "k-means++"用一种特殊的方法选定初始质心从而能加速迭代过程的收敛
           或者给定形如ndarray的初始化质心
    """
    cls = KMeans(n_clusters=4, init='k-means++')
    
    y_hat = cls.fit_predict(data)
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)

    #对data进行旋转处理
    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)
    y_r_hat = cls.fit_predict(data_r)

    #设置字体
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))
    
    #可视化原始数据
    plt.figure(figsize=(9, 10), facecolor='w')
    plt.subplot(421)
    plt.title('原始数据')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    #可视化kmeans++聚类后的数据
    plt.subplot(422)
    plt.title(u'KMeans++聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    #可视化旋转后的数据
    plt.subplot(423)
    plt.title(u'旋转后数据')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data_r, axis=0)
    x1_max, x2_max = np.max(data_r, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    #可视化旋转后的KMeans++聚类
    plt.subplot(424)
    plt.title(u'旋转后KMeans++聚类')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    #可视化方差不相等的数据
    plt.subplot(425)
    plt.title(u'方差不相等数据')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data2, axis=0)
    x1_max, x2_max = np.max(data2, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    #可视化方差不相等的KMeans++聚类
    plt.subplot(426)
    plt.title(u'方差不相等KMeans++聚类')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    #可视化不平衡数据
    plt.subplot(427)
    plt.title(u'数量不相等数据')
    plt.scatter(data3[:, 0], data3[:, 1], s=30, c=y3, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data3, axis=0)
    x1_max, x2_max = np.max(data3, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    #可视化数量不相等的KMeans++聚类
    plt.subplot(428)
    plt.title(u'数量不相等KMeans++聚类')
    plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.tight_layout(2)
    plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)
    plt.subplots_adjust(top=0.92)
    plt.show()
