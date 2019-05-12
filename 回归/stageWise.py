# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:43:27 2019
前向逐步回归
@author: Kylin
"""
import numpy as np
from OLSregression import loadDataSet

def regularize(xMat):
    """
    对xMat按照列进行标准化处理
    输入：原特征矩阵xMat
    输出：标准化特征矩阵inMat
    """
    inMat = xMat.copy() #为了防止修改原来的数据，先进行copy
    inMeans = np.mean(inMat,0) #按列计算均值
    inVar = np.var(inMat,0) #按列计算方差
    inMat = (inMat - inMeans) / inVar #得到标准化数据
    return inMat

def rssError(yArr,yHatArr): 
    #计算真实值与预测值之间的误差平方和，注意yArr和yHatArr都是同维的数组
    return ((yArr-yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    """
    逐步线性回归算法
    输入：数据xArr，目标yArr，每次迭代调整的步长eps和迭代次数numIt
    """
    #1. 首先对数据进行标准化处理
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = np.shape(xMat) #得到样本数m,特征数n
    returnMat = np.zeros((numIt,n)) #产生测试所用的numIt*n阶特征数矩阵，
    ws = np.zeros((n,1)) #产生n*1阶的权重矩阵
    wsTest = ws.copy() #为实现贪心算法copy的两份副本
    wsMax = ws.copy()
    for i in range(numIt):
        """迭代numIt次
        对每轮迭代进行如下的操作：
        """
        print("第%d次迭代的权值数组如下:"%i)
        print(ws.T)
        lowestError = np.inf #将lowestError设置为正无穷
        for j in range(n):
            #对每个特征进行如下的操作，计算减小或增加该特征对误差的影响
            for sign in [-1,1]:
                #改变系数，得到一个新的权重数组
                wsTest = ws.copy() #copy原数组
                wsTest[j] += eps*sign #加或减步长
                yTest = xMat*wsTest #查看使用新权值得到的预测值
                rssE = rssError(yMat.A,yTest.A)#计算误差
                if rssE < lowestError:
                    #如果误差小于当前的最小误差
                    lowestError = rssE
                    #就将wsTest设置为当前的权值
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

if __name__ == "__main__":
    xArr, yArr = loadDataSet('abalone.txt')
    regression = stageWise(xArr, yArr, 0.001, 200)
    