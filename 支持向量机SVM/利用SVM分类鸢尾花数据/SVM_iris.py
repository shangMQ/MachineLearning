#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


# 'sepal length', 'sepal width', 'petal length', 'petal width'
#已知鸢尾花数据有四个特征
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'


def show_accuracy(predict, real, tip):
    """
    计算准确率函数
    参数：
    predict:预测类标
    real:真实类标
    tip:测试集or训练集
    """
    acc = predict.ravel() == real.ravel()
    print(tip + '正确率：', np.mean(acc))


if __name__ == "__main__":
    #设置图像的中文字体
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    
    #1. 利用pandas加载数据集
    path = 'iris.data'  # 数据文件路径
    data = pd.read_csv(path, names=["sepal-length", "sepal-width",
                                     "petal-length", "petal-width", "label"])
    #为了便于处理数据，将label标记为int型数据
    data["label"] = pd.Categorical(data["label"]).codes
    
    #2. 选择前两列作为特征数组，并将类标从字符串转换为数字
    x = np.array(data[["sepal-length", "sepal-width"]])
    y = np.array(data["label"])
    #划分训练集和测试机，60%用作训练，40%留作测试
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

    #3. 利用训练集的数据拟合分类器
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    #4. 查看准确率
    print(clf.score(x_train, y_train))  # 精度
    y_hat = clf.predict(x_train)
    show_accuracy(y_hat, y_train, '训练集')
    print(clf.score(x_test, y_test))
    y_hat = clf.predict(x_test)
    show_accuracy(y_hat, y_test, '测试集')

    #5. 画图
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    Z = clf.decision_function(grid_test)    # 样本到决策面的距离
    print("每个样本点到其他三个类别的间隔边界的距离：")
    print(Z) #Z值是每个样本点到其他三个类的距离
    grid_hat = clf.predict(grid_test) # 预测分类值
    print("预测值：")
    print(grid_hat)
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

    plt.scatter(x[:, 0], x[:, 1], s=50, c=y, edgecolors='k', cmap=cm_dark)      # 样本
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, c=y_test, edgecolors='k', cmap=cm_dark) # 圈中测试集样本
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
    plt.grid()
    plt.show()
    