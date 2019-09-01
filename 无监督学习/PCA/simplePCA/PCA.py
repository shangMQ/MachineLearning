# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 09:57:49 2019
利用numpy实现pca
@author: Kylin
"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

def loadData(filename, delim="\t"):
    """
    加载数据集
    参数：
        filename:文件名
        delim:分隔符，默认为制表符"\t"
    返回值:
        dataMat:数据矩阵
    """
    f = open(filename, 'r')
    data = []
    #此时获取的数据均为字符串格式,需要将其转换为float格式
    for line in f.readlines():
        numstring = line.strip().split(delim)
        subdata = []
        for num in map(float, numstring):
            subdata.append(num)
        data.append(subdata)
    
    #将数据转换为矩阵形式
    dataMat = np.mat(data)
    return dataMat

def pca(dataMat, topNfeat=1):
    """
    PCA主函数
    参数：
        dataMat:数据矩阵
        topNfeat:应用的N个特征，默认为1
    返回值：
        lowDDataMat:降维之后的数据集
        reConMat:重构数据集
    """
    #1. 减去平均值
    meanVals = np.mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals

    #2. 计算协方差矩阵
    covMat = np.mat(np.cov(meanRemoved, rowvar = 0))
    
    #3. 计算协方差矩阵的特征值和特征向量
    eigVals, eigVects = linalg.eig(covMat)
    
    #4. 将特征值从大到小排序, 保留下标，保留最前面的topNfeat个特征值
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]

    #5. 保留topNfeat个特征向量
    redEigVects = eigVects[:, eigValInd]
    print(redEigVects)
    
    #6. 将数据转换到新的空间中
    lowDDataMat = meanRemoved * redEigVects
    
    #7. 重构数据集
    reConMat = (lowDDataMat * redEigVects.T) + meanVals
    
    return lowDDataMat, reConMat

def printFig(dataMat, reConMat):
    """
    绘制原数据图像和重构数据图像
    参数:
        dataMat:原始数据集
        reConMat:重构数据集
    """
    fig = plt.figure("原始数据与重构数据对比")
    ax = fig.add_subplot(121)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0])
    ax.set_title("Original data")
    ax2 = fig.add_subplot(122)
    ax2.scatter(reConMat[:,0].flatten().A[0], reConMat[:,1].flatten().A[0])
    ax2.set_title("Reconstruction data")
   
def printFig2(dataMat, reConMat):
    """
    绘制原数据图像和双特征重构数据图像
    参数:
        dataMat:原始数据集
        reConMat:重构数据集
    """
    fig2 = plt.figure("原始数据与重构数据对比(双特征)")
    plt.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], c='r', label="Original", s=30)
    plt.scatter(reConMat[:,0].flatten().A[0], reConMat[:,1].flatten().A[0], c='b', label="Reconstruction", alpha=0.5, s=45)
    plt.legend()
    plt.title("Double Features Restore Initial Data")

if __name__ == "__main__":
    #加载原始数据
    filename = "testSet.txt"
    dataMat = loadData(filename)
    #利用一个特征
    print("-----Singular Feature------")
    lowDDataMat, reConMat = pca(dataMat,1)
    print("降维后的数据集：", lowDDataMat)
    print("重构数据集：", reConMat)
    printFig(dataMat, reConMat)
    
    #利用两个特征，还原数据集
    print("-----Double Features------")
    lowDDataMat, reConMat = pca(dataMat,2)
    print("降维后的数据集：", lowDDataMat)
    print("重构数据集：", reConMat)
    printFig2(dataMat, reConMat)