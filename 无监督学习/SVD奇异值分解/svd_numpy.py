# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:43:33 2019
SVD奇异值分解的初次尝试
@author: Kylin
"""
import numpy as np

#1. 构建一个矩阵
data = np.mat([[1,1],[7,7]])

#2. 利用linalg中的svd函数计算矩阵的奇异值分解
U, sigma, VT = np.linalg.svd(data)

#3. 输出矩阵的奇异值分解
print("U:")
print(U)
print("sigma:")
print(sigma)
print("VT:")
print(VT)
