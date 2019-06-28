# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:28:16 2019
感知机模型的简单实现
已知训练数据集，正实例点为x1=(3,3),x2=(4,3),负实例点是x3=(1,1)
试用感知机学习算法的原始形式求感知机模型
@author: 尚梦琦
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

def showData(X, y):
    plt.scatter(X[:,0], X[:,1], color='r', label="Initial Data")
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.legend()
    plt.title("感知器的简单实现")
    
def calcLoss(x, y, w, b):
    """
    计算特定样本的损失函数
    参数：样本特征x,样本类标y, w, 超平面截距b
    输出：损失函数值
    """
    loss = float(- y * ((x * w) + b))
    return loss

def updateParameters(x, y, w, b, η):
    """
    更新感知机模型的相关参数
    参数：样本特征x,样本类标y, w, 超平面截距b, 学习率η
    返回值：修改后的参数值
    """
    #利用SGD原理更新w
    w += η * (x * y)
    b += η * y
    
    return w, b
    
def randomSelect(X, m):
    i = random.randint(0, m-1)
    return i

def getLoss(X, y, w, b, m):
    sumloss = 0
    for i in range(m):
        loss =  calcLoss(X[i], y[i], w, b)
        if loss < 0:
            sumloss += loss
    return sumloss
            
def getParameters(X, y, η, n):
    """
    获取感知机模型的相关参数
    参数：特征数组X, 类别数组y, 学习率η, 迭代次数n
    返回值：w, b
    """
    #m是样本数，n是样本特征数
    m, n = np.shape(X)
    X = np.mat(X)
    y = np.mat(y).T
    #初始化w为一个值为0的二维数组, b的初值为0
    w = np.mat(np.zeros((n,1)))
    b = 0
    t = 0
    loss = 0
    
    while (t < n) and (loss <= 1e-4):
        print("*"*5, "当前第{:}次迭代".format(t), "*"*5)
        #随机选择一个样本
        i = randomSelect(X, m)
        print(i)
        if calcLoss(X[i], y[i], w, b) != y[i]:
            #如果当前样本未能正确分类
            print("第%d个样本可以用于更新"%i)
            w, b = updateParameters(X[i], y[i], w, b, η)
            
        loss = getLoss(X, y, w, b, m)
        t += 1
    
    return w, b

#设置中文字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#加载数据集
X = np.array([[3, 3],
              [4, 3],
              [1, 1]])
y = np.array([1, 1, -1])

#可视化原始数据集
showData(X, y)

#得到感知器模型的参数w，b
w, b = getParameters(X, y, 1, 10)
print(w, b)



