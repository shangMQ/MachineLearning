# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:28:16 2019
感知机模型的简单实现
已知训练数据集，正实例点为x1=(3,3),x2=(4,3),负实例点是x3=(1,1)
试用感知机学习算法的原始形式求感知机模型
注意：由于每次选择用于更新w和b的点是随机的，所以最后的结果也是随机的。
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
    loss = float(y * ((x * w) + b))
    return loss

def updateParameters(x, y, w, b, η):
    """
    更新感知机模型的相关参数
    参数：样本特征x,样本类标y, w, 超平面截距b, 学习率η
    返回值：修改后的参数值
    """
    #利用SGD原理更新w
    w += (η * y * x).T
    b += η * float(y)
    print("w =", w, "b =", b)
    return w, b
    
def randomSelect(X, m):
    i = random.randint(0, m-1)
    return i

def getLoss(X, y, w, b, m):
    sumloss = 0
    for i in range(m):
        loss =  calcLoss(X[i], y[i], w, b)
        if loss < 0:
            sumloss += -loss
    return sumloss

            
def getParameters(X, y, η):
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
    flag = True
    
    while getLoss(X, y, w, b, m) != 0 or flag:
        flag = False
        print("*"*5, "当前第{:}次迭代".format(t), "*"*5)
        #随机选择一个样本
        i = randomSelect(X, m)
        print(i)
        if calcLoss(X[i], y[i], w, b) <= 0:
            #如果当前样本未能正确分类
            print("第%d个样本可以用于更新"%i)
            w, b = updateParameters(X[i], y[i], w, b, η)
            
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
w, b = getParameters(X, y, 1)
print("w =", w)
print("b =", b)

#绘制分隔超平面
t = np.linspace(0, 5, 100)
w = np.array(w)
if w[1] == 0:
    f = -b / w[0]
    #使用plt.vlines(x, ymin, ymax)绘制垂线
    plt.vlines(f, 0, 4, colors = "b", linestyles = "-", label="分隔超平面")
else:
    f = (-b - w[0] * t) / w[1]
    plt.plot(t, f, label="分隔超平面")
plt.legend()