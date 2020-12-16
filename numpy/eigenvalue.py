# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
矩阵特征值与特征向量的计算
"""
import numpy as np

# 1. 输入对角阵
A = np.diag((1, 2, 3))
print(A)

# 2. 使用numpy的linear algebra线性代数包，计算特征值与特征向量
a, x = np.linalg.eig(A)

# 其中，a中存放的是特征值，array数组中
print(a)
# x中存放的是特征向量，也是在array数组中
print(x)