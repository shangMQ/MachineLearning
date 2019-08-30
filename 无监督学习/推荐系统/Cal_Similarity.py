# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:21:44 2019
相似度计算
@author: Kylin
"""
import numpy as np
from numpy import linalg

def loadExData():
    data = np.mat([[1, 1, 1, 0, 0],
                   [2, 2, 2, 0, 0],
                   [1, 1, 1, 0, 0],
                   [5, 5, 5, 0, 0],
                   [1, 1, 0, 2, 2],
                   [0, 0, 0, 3, 3],
                   [0, 0, 0, 1, 1]])
    return data

def euclidSim(inA, inB):
    #利用欧式距离计算相似度
    return 1.0 / (1.0 + linalg.norm(inA - inB))

def pearsSim(inA, inB):
    #利用皮尔森相关系数计算相似度
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
    #利用余弦计算相似度
    n = inA.shape[1] #n是inA和inB的特征数
    inA = np.array(inA).reshape(n,)
    inB = np.array(inB).reshape(n,)
    num = np.dot(inA,inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num/denom)

if __name__ == "__main__":
    data = loadExData()
    x, y = data[0], data[4]
    print("第0列和第4列的欧氏距离为:", euclidSim(x, y))
    print("第0列和第4列的皮尔森相关系数为:", pearsSim(x, y))
    print("第0列和第4列的余弦相似度为:", cosSim(x,y))
    



