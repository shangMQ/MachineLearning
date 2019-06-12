# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:07:03 2019
简易版SMO算法的实现
与SMO算法中的外循环确定要优化的最佳α对，简化版跳过这一步骤，首先在数据集上遍历每一个α，
然后在剩下的α集合中随机选择另一个α，从而构建α对。
@author: Kylin
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def loadDataSet(filename):
    """
    加载数据集
    参数：文件名
    """
    dataList = []
    labelList = []
    fr = open(filename)
    
    for line in fr:
        lineArr = line.strip().split('\t')
        dataList.append([float(lineArr[0]), float(lineArr[1])])
        labelList.append(float(lineArr[2]))
    
    return dataList, labelList

def selectJrand(i, m):
    """
    在某个区间范围内随机选择一个整数
    参数：α的下标i，α的总个数m
    返回值：为了与i构成α对而随机选择一个下标j
    """
    mlist = [j for j in range(m)]
    del mlist[i]
    index = int(random.uniform(0, m-1))
    return mlist[index]

def showData(dataArr, labelArr):
    """
    可视化数据
    """
    m = len(dataArr)
    positive = []
    negative = []
    for i in range(m):
        if labelArr[i] == 1:
            positive.append(dataArr[i])
        else:
            negative.append(dataArr[i])
    
    positiveArr = np.array(positive)
    negativeArr = np.array(negative)
    fig = plt.figure("数据可视化")
    plt.scatter(positiveArr[:, 0].ravel(), positiveArr[:, 1].ravel(), c='b', label="positive")
    plt.scatter(negativeArr[:, 0].ravel(), negativeArr[:, 1].ravel(), c='r', label="negative")
    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.legend()
    plt.title("DataSet")
    plt.show()

def clipAlpha(aj, H, L):
    """
    当aj的数值太大时进行调整
    参数：aj，上界H，下界L
    返回值：调整后的aj
    """
    if aj > H:
        #大于上界
        aj = H
    elif aj < L:
        #低于下界
        aj = L
    return aj

def simple(dataList, labelList, C, toler, maxIter):
    """
    简易SMO
    参数：
        dataList:特征数据
        labelList:类标数据
        C:常数
        toler:容错率
        maxIter:最大迭代次数
    """
    #为了便于处理，先将数据集转换成numpy矩阵
    dataMatrix = np.mat(dataList)
    labelMartix = np.mat(labelList).T
    m, n = np.shape(dataMatrix) #m为样本总数，n为特征数
    b = 0 #初始化b为0
    alphas = np.mat(np.zeros((m,1))) #初始化α为0
    iter = 0 #存储没有任何α改变的情况下遍历数据集的次数，当达到maxIter时，结束函数
    
    while (iter < maxIter):
        alphaPairsChanged = 0 #记录α是否已经进行优化
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMartix).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - labelMatrix[i]
            


if __name__ == "__main__":
    #1. 加载数据集
    filename = "testSet.txt"
    dataArr, labelArr = loadDataSet(filename)
    showData(dataArr, labelArr)    
    #2. 创建一个α向量并将其初始化为0向量
    #3. 当迭代次数小于最大迭代次数时（外循环）
        #4. 对数据集中的每个数据向量（内循环）
            #5. 如果该数据向量可以被优化
                #6. 随机选择另外一个数据向量
                #7. 同时优化这两个向量
                #8. 如果两个向量都不能被优化，退出内循环
        #如果所有向量都没被优化，增加迭代数目，继续下一次循环
        
    
    
