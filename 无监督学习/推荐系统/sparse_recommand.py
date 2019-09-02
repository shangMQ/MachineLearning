# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:00:20 2019
使用svd挑选稀疏矩阵特征数的食物推荐系统
@author: Kylin
"""
import numpy as np
from numpy import linalg as la
from foodrecom import cosSim, recommend

def loadData():
    #实际上，通常会遇到的是稀疏矩阵
    data = np.mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                   [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                   [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                   [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                   [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                   [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                   [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                   [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                   [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                   [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                   [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])
    return data

def nintyPercent(dataMat):
    """
    通过svd分解，求出包含90%信息所需的特征数并将物品转换到低维空间中
    参数：数据矩阵
    返回值：转换到低维空间后的数据矩阵
    """
    U, sigma, VT = la.svd(dataMat)
    n = len(sigma) #奇异值个数
    sigma2 = sigma**2 #便于计算，这里对sigma进行平方处理
    energy = sum(sigma2) * 0.9 #计算90%信息量
    for i in range(1, n+1):
        subenergy = sum(sigma2[:i])
        if subenergy >= energy:
            break
    #利用奇异值构建对角矩阵
    sig = np.mat(np.eye(i) * sigma[:i])
    #利用U矩阵将物品转换到低维空间中(sig.I计算的是sig的逆矩阵)
    xformedItems = dataMat.T * U[:,:i] * sig.I
    return xformedItems
    

def svdEst(dataMat, user, item, similarMeasurment = cosSim):
    """
    计算在给定相似度计算方法下，用户对物品的估计评分值
    参数：
    dataMat：数据矩阵
    user：用户编号
    similarMeasurment：相似度计算方法,默认使用余弦相似度计算
    item：物品编号
    返回值：最终评分
    """
    n = dataMat.shape[1] #获取菜品种类个数
    simTotal = 0 #初始化两个变量值
    ratSimTotal = 0
    
    #转换到低维空间后的数据矩阵
    xformedItems = nintyPercent(dataMat)
    
    for j in range(n):
        #遍历n个菜品，j是每一个菜品
        userRating = dataMat[user, j] #获取相应用户对某个菜品的评分
        
        if userRating == 0 or j == item:
            #用户未对该菜品进行评价，直接跳过
            continue
        similarity = similarMeasurment(xformedItems[item,:].T, xformedItems[j,:].T)
        print('the %d and %d similarity is: %f'%(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        return ratSimTotal/simTotal

if __name__ == "__main__":
    dataMat = loadData()
    user = int(input("想要推荐的用户id："))
    recommend(dataMat, user, similarMeas=cosSim, estMethod=svdEst)
