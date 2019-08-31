# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:11:57 2019
基于svd的图像压缩
@author: Kylin
"""
import numpy as np
from numpy import linalg as la

def loadData():
    """
    加载手写数字文件,将其转换为数字矩阵
    返回值：
        dataMat:手写数字矩阵
    """
    filename = "handwrite.txt"
    f = open(filename, 'r')
    datalist = []
    
    for line in f.readlines():
        line = line.strip()
        sub = [int(line[i]) for i in range(len(line))]
        datalist.append(sub)
    
    f.close()
    dataMat = np.mat(datalist)
    
    return dataMat

def printMat(inMat, thresh=0.8):
    """
    打印矩阵，由于包含浮点数，设定阈值
    当数据大于阈值时，打印1，否则打印0
    参数:
        inMat:数据矩阵
        thresh:阈值
    """
    m, n = inMat.shape
    for i in range(m):
        for j in range(n):
            if float(inMat[i, j]) > thresh:
                print(1, end="")
            else:
                print(0, end="")
        print()

def imgCompress(numSV=3, thresh=0.8):
    """
    基于任意给定的奇异值数目来重构图像
    参数:
        numSV:奇异值数目
        thresh:阈值
    """
    #加载原始数据矩阵
    dataMat = loadData()
    print("-"*12,"原图像","-"*12)
    printMat(dataMat, thresh)
    
    #对原始数据进行奇异值分解
    U, sigma, VT = la.svd(dataMat)
    
    #重构奇异值矩阵
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k,k] = sigma[k]
    
    #重构后的手写数字矩阵
    reconMat = U[:,:numSV] * SigRecon*VT[:numSV,:]
    print("-"*12,"重构图像","-"*12)
    printMat(reconMat)
    

if __name__ == "__main__":
    imgCompress(2)
