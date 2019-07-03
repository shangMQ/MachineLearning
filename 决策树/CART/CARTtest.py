# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:22:18 2019
利用CART，使用平方误差损失准则生成一个二叉回归树
@author: Kylin
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    """
    加载数据集
    """
    dataSet = np.array([[1,4.5],
                        [2, 4.75],
                        [3, 4.91],
                        [4, 5.34],
                        [5, 5.8],
                        [6, 7.05],
                        [7, 7.9],
                        [8, 8.23],
                        [9, 8.7],
                        [10, 9]])
    return dataSet

def binSplitDataSet(dataSet, feature, value):
    """
    根据给定特征的值二分数据集
    参数：
        dataSet：数据集
        feature：给定特征
        value: 给定特征值
    """
    left = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    right = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return left, right

def regLeaf(dataSet):
    #返回叶节点的值
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    #计算最后一列的方差，并乘上总样本数
    m = np.shape(dataSet)[0] #m为样本总数
    return np.var(dataSet[:,-1]) * m #返回总方差

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(0.3,1)):
    """
    用最佳的方式切分数据集并生成相应的叶子节点
    参数：
        dataSet：数据集矩阵
        leafType：叶子节点类型
        errType：损失函数
        ops:用户定义参数，用于完成树的构建
    返回值：最佳特征下标，相应的特征值
    """
    tolS = ops[0] #容许的误差下降值
    tolN = ops[1] #切分的最少样本数
    
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: 
        #如果当前的所有样本的值相同，则不需要继续二分数据集，返回None，代表应该建立叶节点
        return None, leafType(dataSet)
    
    m,n = np.shape(dataSet) #得到样本数，和特征数
    #the choice of the best feature is driven by Reduction in RSS error from mean
    
    S = errType(dataSet) #计算总方差
    
    #初始化用于切分数据集的相关数值
    bestS = np.inf#最佳切分误差
    bestIndex = 0 #最佳划分特征索引值
    bestValue = 0 #最佳划分特征值
    
    for featIndex in range(n-1):
        #对每个特征
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            #对每个特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)#利用对应的特征值二分数据集
            
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): 
                #如果样本数小于最小切分样本数，则进行下次循环
                continue
            
            #计算当前切分后的总误差
            newS = errType(mat0) + errType(mat1)
            
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
                
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        #如果切分前总误差-当前切分的最佳误差小于容许的误差下降值，则代表应该建立叶节点
        return None, leafType(dataSet) #exit cond 2
    
    #利用最佳切分特征和相应的值进行二分数据集
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  #exit cond 3
        #如果切分后的样本个数小于用户定义的最小切分样本数，则代表应该建立叶节点
        return None, leafType(dataSet)
    
    #否则返回最佳分类特征下标和相应的特征值
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(0.3,1)):
    #assume dataSet is NumPy Mat so we can array filtering
    """
    创建树
    参数：
        dataSet：数据集矩阵
        leafType：叶子节点类型
        errType：损失函数
        ops:用户定义参数，用于完成树的构建
    返回值：构造的树
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None:
        #没有多余的特征可供选择
        return val 
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #利用给定的特征和值二分数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    #构建左子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    #构建右子树
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def showData(myMat):
    x = (myMat[:,0].ravel()[0]).A.tolist()
    y = (myMat[:,1].ravel()[0]).A.tolist()
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")

if __name__ == "__main__":
    #获取数据
    myDat = loadDataSet()
    myMat = np.mat(myDat)
    #数据可视化
    showData(myMat)
    #构建CART树
    tree = createTree(myMat)
    print(tree)
    
    
    
    
    


