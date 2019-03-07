# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:48:04 2019

@author: Kylin
处理鸢尾花数据
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#将鸢尾花数据集转换为DataFrame对象，并加载到内存中
df = pd.read_csv("iris-data.csv",header=None)

#将标记存储到y中
y = df.iloc[0:100, 4].values

#标准化鸢尾花标记，将山鸢尾标准化为1，变色鸢尾标准化为-1。
normal_y = np.where(y == 'Iris-setosa', 1, -1)

#提取两个特征：萼片长度（第1列）和花瓣长度（第3列）
x = df.iloc[0:100, [0, 2]].values

#绘制散点图
plt.scatter(x[:50,0], x[:50,1], color = 'red', marker = 'o', label = 'Setosa')
plt.scatter(x[50:100,0], x[50:100,1], color = 'blue', marker = 'x', label = 'versicolor')
#横轴是萼片长度，纵轴是花瓣长度。
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
