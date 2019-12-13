# -*- coding: utf-8 -*-
"""
关于numpy中的dot计算
"""
import numpy as np

#1. 第一种情况，假设a和b都是一维向量
a = np.array([1,2,3])
b = np.array([2,1,6])

c = np.dot(a,b)
print("两个一维向量的dot计算是点乘计算，输出结果是一个数")
print(c)

#2. 第二种情况，假设a是二维向量而b是一维向量
a = np.array([[1,2,3],
              [4,5,6]])

b = np.array([2,1,6])

c = np.dot(a,b)
print("二维向量和一维向量的dot计算是按行进行的点乘计算，输出结果是一个向量")
print(c)

#2. 第二种情况，假设a和b都是二维向量
a = np.array([[1,2,3],
              [4,5,6]])

b = a.T

c = np.dot(a,b)
print("二维向量a（m行k列）和b（k行n列）的dot计算是矩阵乘法，输出结果是一个二维矩阵（m行n列）")
print(c)


