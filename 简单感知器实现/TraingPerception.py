# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:48:04 2019

@author: Kylin
基于鸢尾花数据训练感知器模型
"""
import matplotlib.pyplot as plt
from Perception import Perception
from data_treating import x, normal_y

#利用鸢尾花数据来训练感知器模型
ppn = Perception(eta = 0.1, n_iter = 10)
ppn.fit(x, normal_y)

#绘制每次迭代的错误分类数量的折线图
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()






