# -*- coding:utf-8 -*-
"""
鸢尾花多分类降维预测，含决策边界的绘制
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def extend(a, b):
    return 1.05 * a - 0.05 * b, 1.05 * b - 0.05 * a


if __name__ == '__main__':
    # 设置降维方式
    stype = 'chi2'

    # 相关设置
    pd.set_option('display.width', 200)

    # 1. 读取数据
    data = pd.read_csv('iris.data', header=None)

    # 2. 数据预处理
    columns = np.array(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type'])
    # columns = np.array(['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类型'])
    # 重置列名
    data.rename(columns=dict(list(zip(np.arange(5), columns))), inplace=True)
    # 将类别用数字表示
    data['type'] = pd.Categorical(data['type']).codes
    print("data:")
    print(data)
    x = data[columns[:-1]]
    y = data[columns[-1]]

    # 特征降维
    if stype == 'pca':  # 利用PCA降维
        pca = PCA(n_components=2, whiten=True, random_state=0)
        x = pca.fit_transform(x)
        print('各方向方差：', pca.explained_variance_)
        print('方差所占比例：', pca.explained_variance_ratio_)
        x1_label, x2_label = 'component 1', 'component 2'
        title = 'Iris features by PCA'
    else:
        fs = SelectKBest(chi2, k=2)
        # fs = SelectPercentile(chi2, percentile=60)
        fs.fit(x, y)
        idx = fs.get_support(indices=True)
        print('选择的特征索引 ：', idx)
        x = x[columns[idx]]
        x = x.values    # 为下面使用方便，DataFrame转换成ndarray
        x1_label, x2_label = columns[idx]
        title = 'iris features by Filter'

    print(x[:5])

    # 3. 绘制特征散点图
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='o', cmap=cm_dark)
    plt.grid(b=True, ls=':', color='k')
    plt.xlabel(x1_label, fontsize=12)
    plt.ylabel(x2_label, fontsize=12)
    plt.title(title, fontsize=15)
    plt.show()

    # 4.数据预测
    # 划分数据集
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.3)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=True)),
        ('lr', LogisticRegressionCV(Cs=list(np.logspace(-3, 4, 8)), cv=5, fit_intercept=False))
    ])
    model.fit(x, y)
    print('最优参数：', model.get_params('lr')['lr'].C_)
    y_hat = model.predict(x)
    print('训练集精确度：', metrics.accuracy_score(y, y_hat))
    y_test_hat = model.predict(x_test)
    print('测试集精确度：', metrics.accuracy_score(y_test, y_test_hat))

    # 绘制决策边界
    N, M = 500, 500     # 横纵各采样多少个值
    x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max())   # 第0列的范围
    x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max())   # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)   # 测试点
    print(x_show.shape)
    y_hat = model.predict(x_show)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同

    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, edgecolors='k', cmap=cm_dark)  # 样本的显示
    plt.xlabel(x1_label, fontsize=12)
    plt.ylabel(x2_label, fontsize=12)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(b=True, ls=':', color='k')

    # 画各种图
    a = mpl.patches.Wedge(((x1_min+x1_max)/2, (x2_min+x2_max)/2), 1.5, 0, 360, width=0.5, alpha=0.5, color='r')
    plt.gca().add_patch(a)
    # 利用mpatches对象手动生成图例
    patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
              mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
              mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
    plt.legend(handles=patchs, fancybox=True, framealpha=0.8, loc='lower right')
    plt.title('Iris prediction By Logistic Regression Classify', fontsize=15)
    plt.show()
