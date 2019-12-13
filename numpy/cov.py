# -*- coding: utf-8 -*-
"""
Spyder Editor
手动计算协方差矩阵和np.cov计算协方差矩阵
可以发现，结果是一样的
注意：手动计算时一定要除以自由度，而且书上对X的假设是每行代表一个特征，而通常来说，我们假设每行代表一个样本。所以这里的dot计算和书上不太一样
"""

import numpy as np

a = np.array([[4,10,20],
              [3, 7, 17],
              [4,11,23],
              [3, 9, 18]])

means = np.mean(a, axis=0)#求各列的均值

#进行去中心化
a_centralized = a - means

print("各个特征的均值")
print(means)
print("去中心化后的数据")
print(a_centralized)

#协方差矩阵计算时，还要除以samples数-1，进行标准化
cal_cov = np.dot(a_centralized.T,a_centralized) / 3

np_cov = np.cov(a, rowvar=0)

print("手动计算cov")
print(cal_cov)
print("np.cov计算结果")
print(np_cov)


