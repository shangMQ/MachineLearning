# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:26:45 2019
利用sklearn提供的预处理函数对数据进行预处理
@author: Kylin
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sphinx.ext.todo import Todo


def svmModel(X_train, X_test, y_train, y_test, str):
    svm = SVC(C=100, gamma=0.2)
    svm.fit(X_train, y_train)
    resultTrain = svm.score(X_train, y_train)
    resultTest = svm.score(X_test, y_test)
    print("=======使用" + str + "数据拟合模型的结果=====")
    print("训练集精度：{:.2f}".format(resultTrain))
    print("测试集精度：{:.2f}".format(resultTest))


def processingCompare(cancer):
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False

    # 只取前两个特征
    data = cancer["data"]
    features = data[:, :2]
    label = cancer["target"]

    feature1 = features[:, 0]
    feature2 = features[:, 1]

    print(feature1.shape)
    print(feature2.shape)

    # 1. 使用原始数据的结果
    plt.figure("Initial Data")
    plt.scatter(feature1, feature2, edgecolors='k')
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.xlim(-30, 30)
    plt.ylim(-40, 40)
    plt.axvline(c='gray')
    plt.axhline(c='gray')
    plt.title("Initial Data")
    plt.show()

    # 2 使用不同的标准化方法对数据进行处理
    fig = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # 2.1 使用MinMaxScaler
    minMaxScaler = MinMaxScaler()
    minMaxScaler.fit(data)
    featureScaled1 = minMaxScaler.transform(data)
    plt.subplot(2, 2, 1)
    plt.scatter(featureScaled1[:,0], featureScaled1[:,1], edgecolors='k')
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axvline(c='gray')
    plt.axhline(c='gray')
    plt.title("MinMaxScaler range=[0, 1]")

    # 2.2 使用StandardScaler
    standardScaler = StandardScaler()
    standardScaler.fit(data)
    featureScaled2 = standardScaler.transform(data)
    plt.subplot(2, 2, 2)
    plt.scatter(featureScaled2[:, 0], featureScaled2[:, 1], edgecolors='k')
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axvline(c='gray')
    plt.axhline(c='gray')
    plt.title("StandardScaler (miu=0, segma=1)")

    # 2.3 使用RobustScaler
    robustScaler = RobustScaler()
    robustScaler.fit(data)
    featureScaled3 = robustScaler.transform(data)
    plt.subplot(2, 2, 3)
    plt.scatter(featureScaled3[:, 0], featureScaled3[:, 1], edgecolors='k')
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axvline(c='gray')
    plt.axhline(c='gray')
    plt.title("RobustScaler")

    # 2.4 使用Normalizer
    normal = Normalizer()
    normal.fit(data)
    featureScaled4 = normal.transform(data)
    plt.subplot(2, 2, 4)
    plt.scatter(featureScaled4[:, 0], featureScaled4[:, 1], edgecolors='k')
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axvline(c='gray')
    plt.axhline(c='gray')
    plt.title("Normalizer")
    plt.suptitle("Different Scaler Methods Comparison", fontsize=16)
    plt.show()


def main():
    # 1.加载数据
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    # 2. 几种预处理方法的对比
    processingCompare(cancer)

    # 3.利用原始数据直接拟合模型
    print("-" * 10, "使用不同的预处理方式初始化数据对模型预测结果的影响", "-" * 10)
    svmModel(X_train, X_test, y_train, y_test, "原始")

    # 查看利用不同的数据预处理方法对原始数据进行处理后的数据预测情况
    # 3.1 构建MinMaxScaler缩放器
    minMaxscaler= MinMaxScaler()
    minMaxscaler.fit(X_train)

    # 3.1.1变换数据
    X_train_scaled1= minMaxscaler.transform(X_train)
    X_test_scaled1 = minMaxscaler.transform(X_test)

    # 3.1.2拟合模型
    svmModel(X_train_scaled1, X_test_scaled1, y_train, y_test, "MinMaxScaler处理后的")
    #
    #
    # 3.2 构建StandardScaler缩放器
    standardscaler = StandardScaler()
    standardscaler.fit(X_train)

    # 3.2.1变换数据
    X_train_scaled2 = standardscaler.transform(X_train)
    X_test_scaled2 = standardscaler.transform(X_test)

    # 3.2.2拟合模型
    svmModel(X_train_scaled2, X_test_scaled2, y_train, y_test, "StandardScaler处理后的")


if __name__== "__main__":
    main()
