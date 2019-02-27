# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:48:04 2019

@author: Kylin
基于鸢尾花数据训练感知器模型
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Perception import Perception

#将鸢尾花数据集转换为DataFrame对象，并加载到内存中
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                 header = None)

y = df.iloc[0:100, 4].values
normal_y = np.where(y == 'Iris-setosa', 1, -1)
x = df.iloc[0:100, [0, 2]].values
plt.scatter(x[:50,0], x[:50,1], color = 'red', marker = 'o', label = 'Setosa')
plt.scatter(x[50:100,0], x[50:100,1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

ppn = Perception(eta = 0.1, n_iter = 10)
ppn.fit(x, normal_y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()






