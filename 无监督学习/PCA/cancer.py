# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 12:19:43 2019
将PCA应用于cancer数据集并可视化
@author: Kylin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def showDataFeatures(cancer):
    """
    由于特征过多，其他方式不便于查看，只能了解每个特征在两个类别中的分类情况。
    """
    # 对于cancer数据集，0表示恶性肿瘤，1表示良性肿瘤
    malignant = cancer.data[cancer.target == 0] # 恶性肿瘤
    benign = cancer.data[cancer.target == 1] # 良性肿瘤
    
    # 2.根据特征绘制图像, 一共30个特征
    fig, axes = plt.subplots(6, 5, figsize=(10, 8))
    ax = axes.ravel()  # 列出所有子图
    
    # 为每一个特征都创建一个直方图，计算具有某一特征的数据点在特定范围内的出现频率
    for i in range(30):
        _, bins = np.histogram(cancer.data[:, i], bins=50)
        # 将类别为恶性的特征用红色表示
        ax[i].hist(malignant[:, i], bins=bins, color="red", alpha=0.5)
        # 类别为良性的特征用蓝色表示
        ax[i].hist(benign[:, i], bins=bins, color="blue", alpha=0.5)
        ax[i].set_title(cancer.feature_names[i])
        ax[i].set_yticks(())  # 设置y轴坐标刻度值
        ax[i].set_xlabel("Feature magnitude")
        ax[i].set_ylabel("Frequency")
    ax[0].legend(["malignant", "benign"], loc="best")
    plt.tight_layout()  # 自动调整子图间的间隔
    plt.show()


def drawMainFeatures(data, X_pca):
    plt.figure(figsize=(8, 8))
    malignant = X_pca[data.target == 0]
    benign = X_pca[data.target == 1]
    plt.scatter(malignant[:, 0], malignant[:, 1], color="red", marker="^", label="malignant")
    plt.scatter(benign[:, 0], benign[:, 1], color="blue", marker="o", label="benign")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.title("Two main component")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False

    # 1. 加载数据
    cancer = load_breast_cancer()
    print("features shape :", cancer.data.shape)
    
    # 2. 利用直方图查看各个特征中各个类别的分布情况
    showDataFeatures(cancer)

    # 3. 利用StandardScaler标准化数据
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    X_scaled = scaler.transform(cancer.data)

    # 4. 之后可以将标准化后的数据应用PCA变换
    pca = PCA(n_components=2)  # n_components设置主成分个数
    pca.fit(X_scaled)

    # 将数据变换到两个主成分的方向上
    X_pca = pca.transform(X_scaled)

    # 查看变换后的数据shape
    print("orginal shape:{:}".format(X_scaled.shape))
    print("After PCA shape:{:}".format(X_pca.shape))

    # 5. 利用前两个主成分作图
    drawMainFeatures(cancer, X_pca)

    # 6. 查看主成分信息
    # 利用pca.components_可以查看模型的特征向量
    print("PCA component shape:{}".format(pca.components_.shape))
    print("PCA components:\n", pca.components_)

    # 7. 利用热力图将系数可视化
    plt.matshow(pca.components_, cmap="viridis")
    plt.yticks([0, 1], ["first component", "second component"])
    plt.colorbar()
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha="left")
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.title("Feature importance by PCA", y=-1, fontsize=16)
    plt.show()
