# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:43:33 2019
SVD奇异值分解的初次尝试
@author: Kylin
"""
import numpy as np

def loadExData():
    data = np.mat([[1, 1, 1, 0, 0],
                   [2, 2, 2, 0, 0],
                   [1, 1, 1, 0, 0],
                   [5, 5, 5, 0, 0],
                   [1, 1, 0, 2, 2],
                   [0, 0, 0, 3, 3],
                   [0, 0, 0, 1, 1]])
    return data

#1. 获取数据矩阵
data = loadExData()

#2. 利用linalg中的svd函数计算矩阵的奇异值分解
U, sigma, VT = np.linalg.svd(data)

#3. 输出矩阵的奇异值分解
print("U:")
print(U)
print("sigma:")
print(sigma)
print("VT:")
print(VT)

#4. 尝试利用秩为3的sigma求原矩阵的近似矩阵
sigmaMat = np.mat([[sigma[0], 0, 0], [0, sigma[1], 0], [0, 0, sigma[2]]])
SimilarData = U[:, 0:3] * sigmaMat * VT[:3, :]

#5. 为了便于和原矩阵做对比，将float类型转为int类型
similar = np.array(SimilarData).astype(int)
print("近似矩阵:")
print(similar)
print("和原矩阵比较结果：")
print(np.array(data) == similar)