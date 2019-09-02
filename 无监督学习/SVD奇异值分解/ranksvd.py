# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:56:08 2019
四名顾客对六种食品进行评价

@author: 尚梦琦
"""
import numpy as np
import numpy.linalg as la

def loadData():
    dataMat = np.mat([[5, 5, 0, 5],
                      [5, 0, 3, 4],
                      [3, 4, 0, 3],
                      [0, 0, 5, 3],
                      [5, 4, 4, 5],
                      [5, 4, 5, 5]])
    return dataMat

def svd(dataMat, num=2):
    """
    利用前num个特征值重构数据
    参数：
        dataMat:数据矩阵
        num:重构采用的特征值数目
    返回值：
        dataRecon:重构后的数据
    """
    #对原矩阵进行svd分解
    U, sigma, VT = la.svd(dataMat)
    
    #使用前num个特征值
    SigRecon = np.mat(np.zeros((num,num)))
    
    for i in range(num):
        SigRecon[i, i] = sigma[i]
    
    #使用num个特征值重构数据
    dataRecon = U[:,:num] * SigRecon * VT[:num,:]
    
    #将浮点值转换为int型
    dataReconArray = dataRecon.A
    for k in range(len(dataReconArray)):
       dataReconArray[k] =  dataReconArray[k].astype("int")
    
    return dataReconArray

if __name__ == "__main__":
    dataMat = loadData()
    print("-------原数据-------")
    print(dataMat)
    U, sigma, VT = la.svd(dataMat)
    print("------重构数据------")
    dataRecon = svd(dataMat)
    print(dataRecon)