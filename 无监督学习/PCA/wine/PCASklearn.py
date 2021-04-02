# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
使用sklearn实现pca
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def loadData():
    """
    加载数据
    :return: 数据特征，类别
    """
    df_wine = pd.read_csv("wine.data", header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    return X, y, df_wine.columns[1:]


def plot_regions(X, y, classifier, message, resolution=0.02):
    """
    绘制决策边界
    :param X:
    :param y:
    :param classifier:
    :param resolution:
    :return:
    """
    # 设置颜色和标记
    markers = ('s', 'x', 'o', '^', 'v')
    cm_light = ListedColormap(['#EB5A2A', '#96CED2', '#68E244'])
    cm_dark = ListedColormap(['r', 'b', 'g'])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))  # 生成网格采样点
    print(xx1.shape)
    print(xx2.shape)

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)
    print(Z)
    plt.contourf(xx1, xx2, Z, cmap=cm_light)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cm_dark(idx), marker=markers[idx], edgecolor='k', label=cl)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(message)
    plt.legend()
    plt.show()


def main():
    # 1. 加载数据
    X, y, label = loadData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 2. 标准化c
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)

    # 3.导入pca模块
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    lr.fit(X_train_pca, y_train)
    plot_regions(X_train_pca, y_train, lr, "Train")
    plot_regions(X_test_pca, y_test, lr, "Test")


if __name__ == "__main__":
    main()