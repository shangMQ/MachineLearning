# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:37:19 2019
对鸢尾花萼片和花瓣长度的二维数据集决策边界的可视化
@author: Kylin
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from Perception import Perception

def plot_decision_regions(X, y, classifier, resolution=0.02):
    #设置标记和颜色地图
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gary', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #找到两个特征的最大最小值
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1 
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    print(xx1.shape)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print("z.shape=", Z.shape)
    Z = Z.reshape(xx1.shape)
    
    #对于网格数组中每个预测的类以不同的颜色绘制出预测得到的决策区域
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #绘制样本的散点图
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
                    alpha=0.8, c = cmap(idx), marker=markers[idx],
                    label=cl)

#将鸢尾花数据集转换为DataFrame对象，并加载到内存中
df = pd.read_csv("iris-data.csv",header=None)

#将标记存储到y中
y = df.iloc[0:100, 4].values
print(df.head())

#标准化鸢尾花标记，将山鸢尾标准化为1，变色鸢尾标准化为-1。
normal_y = np.where(y == 'Iris-setosa', 1, -1)
print(normal_y)

#提取两个特征：萼片长度（第1列）和花瓣长度（第3列）
X = df.iloc[0:100, [0, 2]].values

ppn = Perception(eta = 0.1, n_iter = 10)
ppn.fit(X, normal_y)
plot_decision_regions(X, normal_y, ppn, 0.02)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('Decision regions plot')
plt.legend(loc="upper left")
