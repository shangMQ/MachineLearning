# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
DPGMM(Dirichlet Process Gaussian Mixture Model)
DPGMM代表狄利克雷过程高斯混合模型，它是一个无限混合模型，狄利克雷过程作为聚类数量的先验分布。
在实践中，近似推理算法使用具有固定最大components分量数的截断分布，但实际使用的分量数取决于数据。
此类允许在具有可变数量的components（小于截断参数 n_components）的高斯混合模型的参数上轻松有效地推断近似后验分布。
"""


import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import scipy as sp
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def expand(a, b, rate=0.05):
    d = (b - a) * rate
    return a-d, b+d


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    # 生成数据
    np.random.seed(0)
    cov1 = np.diag((1, 2))
    N1 = 500
    N2 = 300
    N = N1 + N2
    x1 = np.random.multivariate_normal(mean=(3, 2), cov=cov1, size=N1)
    # 对数据进行旋转
    m = np.array(((1, 1), (1, 3)))
    x1 = x1.dot(m)
    x2 = np.random.multivariate_normal(mean=(-1, 10), cov=cov1, size=N2)
    x = np.vstack((x1, x2))
    y = np.array([0]*N1 + [1]*N2)

    # 假定设定的components不合适
    n_components = 3

    # 绘图使用
    colors = '#A0FFA0', '#2090E0', '#FF8080'
    cm = mpl.colors.ListedColormap(colors)
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)

    plt.figure(figsize=(6, 6), facecolor='w')
    plt.suptitle('GMM/DPGMM比较', fontsize=15)

    ax = plt.subplot(211)
    # 使用高斯混合模型预测
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(x)
    centers = gmm.means_
    covs = gmm.covariances_
    print('GMM均值 = \n', centers)
    print('GMM方差 = \n', covs)
    y_hat = gmm.predict(x)

    # 绘制网格数据
    grid_hat = gmm.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm)
    plt.scatter(x[:, 0], x[:, 1], s=20, c=y, cmap=cm, marker='o', edgecolors='#202020')

    clrs = list('rgbmy')
    for i, (center, cov) in enumerate(zip(centers, covs)):
        # 获得实对称阵（协方差）的特征值和特征向量
        value, vector = sp.linalg.eigh(cov)
        # 特征值分别对应椭圆的长半轴和短半轴
        width, height = value[0], value[1]
        # 对特征向量做标准化，scipy.linalg.norm获得范数
        v = vector[0] / sp.linalg.norm(vector[0])
        # 弧度转角度
        angle = 180 * np.arctan(v[1] / v[0]) / np.pi
        e = Ellipse(xy=center, width=width, height=height,
                    angle=angle, color=clrs[i], alpha=0.5, clip_box = ax.bbox)
        ax.add_artist(e)

    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.title('GMM', fontsize=15)
    plt.grid()

    # DPGMM
    dpgmm = BayesianGaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000, n_init=5,
                                    weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=0.1)
    dpgmm.fit(x)
    centers = dpgmm.means_
    covs = dpgmm.covariances_
    print('DPGMM均值 = \n', centers)
    print('DPGMM方差 = \n', covs)
    y_hat = dpgmm.predict(x)
    print(y_hat)

    ax = plt.subplot(212)
    grid_hat = dpgmm.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm)
    plt.scatter(x[:, 0], x[:, 1], s=20, c=y, cmap=cm, marker='o', edgecolors='#202020')

    for i, cc in enumerate(zip(centers, covs)):
        # 虽然得到三个成分，但是数据完全由前两个成分拟合，
        # 而属于第三个成分的元素不存在，因此跳过
        if i not in y_hat:
            continue
        center, cov = cc
        value, vector = sp.linalg.eigh(cov)
        width, height = value[0], value[1]
        v = vector[0] / sp.linalg.norm(vector[0])
        angle = 180 * np.arctan(v[1] / v[0]) / np.pi
        e = Ellipse(xy=center, width=width, height=height,
                    angle=angle, color='m', alpha=0.5, clip_box=ax.bbox)
        ax.add_artist(e)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.title('DPGMM', fontsize=15)
    plt.show()
