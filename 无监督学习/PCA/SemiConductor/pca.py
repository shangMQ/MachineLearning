# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:55:22 2019
利用PCA对半导体数据进行降维
@author: Kylin
"""
import numpy as np
import numpy.linalg as la

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

def replaceNanWithMean():
    """
    利用均值替换为NAN的数据
    返回值：
        填充后的数据
    """
    dataMat = loadData("secom.data", " ")
    FeatureNum = dataMat.shape[1]
    
    for i in range(FeatureNum):
        #均值(数组(非零(非nan(第i个特征数组))))
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:,i].A))[0],i])
        
        #利用均值填充
        dataMat[np.nonzero(np.isnan(dataMat[:,i].A))[0],i] = meanVal
    
    return dataMat

def pca(dataMat, topNfeat):
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
    eigVals, eigVects = la.eig(covMat)
    
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
    
if __name__ == "__main__":
    dataMat = replaceNanWithMean()
    MaxFeatures = 6
    lowDDataMat, reConMat = pca(dataMat, MaxFeatures)
    print(lowDDataMat)
