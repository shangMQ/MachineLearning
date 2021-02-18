# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:56:23 2019
基于正规方程的方法利用最小二乘法(ordinary least squares)计算数据的最佳拟合直线
@author: Kylin
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def loadDataSet(filename):
    """
    根据filename加载数据集
    其中，数据集的前两项数据为特征，最后一项数据为目标值
    """
    f = open(filename)
    featurenum = len(f.readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    for line in f.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(featurenum):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
                
    f.close()
    return dataMat, labelMat


def standRegression(XArr, yArr):
    """
    计算正规方程的值，同时对当特征矩阵为奇异矩阵时做出判断
    输入：特征数组，目标值数组
    输出：系数矩阵
    """
    # 利用np.mat()方法将数组转化为矩阵形式
    XMat = np.mat(XArr)
    yMat = np.mat(yArr).T
    XTX = XMat.T * XMat
    # np.linalg.det()计算行列式
    if np.linalg.det(XTX) == 0.0:
        print("该矩阵为奇异矩阵，无法计算X.T*X的逆矩阵")
        return
    weights = XTX.I *(XMat.T * yMat)
    return weights


def getPredict(XArr, weights):
    """
    根据回归系数，计算预测值
    输入：XArr特征数组，weights系数矩阵
    输出：y的预测值(矩阵格式)
    """
    XMat = np.mat(XArr)
    predicty = XMat * weights
    return predicty


def plot(XArr, yArr, predicty):
    """
    绘制散点图和拟合后的直线
    输入值：XArr列表，yArr列表，predicty矩阵
    """
    # 首先对数据进行排序，便于画图
    X = np.array(XArr)
    y = np.array(yArr)
    predicty = np.array(predicty)
    feature2 = X[:,1]
    index = y.argsort()
    print(index)
    featureData = X[index]
    target = y[index]

    # 绘制数据点
    fig1 = plt.figure()
    plt.scatter(featureData[:,1], target)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Data Show")
    plt.show()

    # 绘制预测值与真实值的对比
    fig2 = plt.figure()
    predict = predicty[index]
    plt.scatter(featureData[:,1], target, color="b", label="Target")
    plt.plot(featureData[:,1], predict, color='r', label="Predict")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()


def calculateCorrcoef(predicty, yArr):
    """
    计算预测值与真实值之间的相关系数
    输入：预测值矩阵，真实值列表
    """
    yMat = np.mat(yArr)
    # 利用np.corrcoef(yEstimate, y)计算相关系数
    coef = np.corrcoef(predicty.T, yMat)
    print(coef)


if __name__ == "__main__":
    # 设置图像的标题字体
    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False

    # 1. 读取数据
    XArr, yArr = loadDataSet("ex0.txt")

    # 2. 正规方程计算权重
    weights = standRegression(np.array(XArr), np.array(yArr))
    print("weights = ", weights)
    w = np.array(weights)
    print("y = {:.2f} + {:.2f}x".format(w[0][0], w[1][0]))

    # 3. 预测
    predicty = getPredict(XArr, weights)
    print("预测结果 = ", predicty)

    # 4. 计算预测值与真实值的匹配程度
    plot(XArr, yArr, predicty)
    print("预测值与真实值之间的相关系数为：")
    calculateCorrcoef(predicty, yArr)
