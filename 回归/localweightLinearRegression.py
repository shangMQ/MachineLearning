# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:57:36 2019
实现局部加权线性回归（LWLR）
@author: Kylin
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from OLSregression import loadDataSet

def lwlr(testPoint, xArr, yArr, k = 1.0):
    #返回对回归系数的一个估计
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0] #有多少个样本点
    weights = np.mat(np.eye(m)) #m阶单位矩阵
    #利用高斯核函数来
    for j in range(m):
        #通过循环计算每个样本点对应的权重值：
        diffMat = testPoint - xMat[j,:]
        #随着样本点与待预测点距离的递增，权重将以指数级衰减。
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    
    xTx = xMat.T*(weights * xMat)
    
    if np.linalg.det(xTx) == 0:
        print("该矩阵是奇异矩阵，无法计算逆矩阵")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    #给定x空间的任意一点，计算对应的预测值yHat
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def plot(xArr, yArr, yHat, k = 1.0):
    """
    绘制预测值与真实值的对比图，注意要对预测值进行排序
    """
    titlestr = "k={:}时的拟合情况图".format(k)
    yarray = np.array(yHat)
    xarray = np.array(xArr)
    srtInd = xarray[:,1].argsort(0)
    xSort = xarray[srtInd]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yarray[srtInd], color='red', lw=2)
    ax.scatter(xarray[:,1], yArr)
    ax.set_title(titlestr)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

if __name__ == "__main__":
    #设置图像的标题字体
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    
    xArr, yArr = loadDataSet("ex0.txt")
    #得到数据集里的所有点的估计，调用lwlrTest()函数
    k = [1, 0.01, 0.003]
    for i in range(len(k)):
        yHat = lwlrTest(xArr, xArr, yArr, k[i])
        plot(xArr, yArr, yHat, k[i])
    
        

