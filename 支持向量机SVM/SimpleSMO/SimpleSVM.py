# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:07:03 2019
简易版SMO算法的实现
与SMO算法中的外循环确定要优化的最佳α对，简化版跳过这一步骤，首先在数据集上遍历每一个α，
然后在剩下的α集合中随机选择另一个α，从而构建α对。
@author: Kylin
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def loadDataSet(filename):
    """
    加载数据集
    参数：文件名
    """
    dataList = []
    labelList = []
    fr = open(filename)
    
    for line in fr:
        lineArr = line.strip().split('\t')
        dataList.append([float(lineArr[0]), float(lineArr[1])])
        labelList.append(float(lineArr[2]))
    
    return dataList, labelList

def selectJrand(i, m):
    """
    在某个区间范围内随机选择一个整数
    参数：α的下标i，α的总个数m
    返回值：为了与i构成α对而随机选择一个下标j
    """
    mlist = [j for j in range(m)]
    del mlist[i]
    index = int(random.uniform(0, m-1))
    return mlist[index]

def showData(dataArr, labelArr, fig1):
    """
    可视化数据
    """
    m = len(dataArr)
    positive = []
    negative = []
    for i in range(m):
        if labelArr[i] == 1:
            positive.append(dataArr[i])
        else:
            negative.append(dataArr[i])
    
    positiveArr = np.array(positive)
    negativeArr = np.array(negative)

    plt.scatter(positiveArr[:, 0].ravel(), positiveArr[:, 1].ravel(), c='b', label="positive")
    plt.scatter(negativeArr[:, 0].ravel(), negativeArr[:, 1].ravel(), c='r', label="negative")
    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.legend()
    plt.title("DataSet")
    plt.show()

def clipAlpha(aj, H, L):
    """
    当aj的数值太大时进行调整
    参数：aj，上界H，下界L
    返回值：调整后的aj
    """
    if aj > H:
        #大于上界
        aj = H
    elif aj < L:
        #低于下界
        aj = L
    return aj

def simple(dataList, labelList, C, toler, maxIter):
    """
    简易SMO
    参数：
        dataList:特征数据
        labelList:类标数据
        C:常数
        toler:容错率
        maxIter:最大迭代次数
    """
    #为了便于处理，先将数据集转换成numpy矩阵
    dataMatrix = np.mat(dataList)
    labelMatrix = np.mat(labelList).T
    m, n = np.shape(dataMatrix) #m为样本总数，n为特征数
    b = 0 #初始化b为0
    alphas = np.mat(np.zeros((m,1))) #创建一个α向量并将其初始化为0向量
    iter = 0 #存储没有任何α改变的情况下遍历数据集的次数，当达到maxIter时，结束函数
    
    while (iter < maxIter):
        #当迭代次数小于最大迭代次数时（外循环）
        alphaPairsChanged = 0 #记录α是否已经进行优化
        
        for i in range(m):
            #对数据集中的每个数据向量（内循环）
            #计算误差
            fXi = float(np.multiply(alphas, labelMatrix).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - labelMatrix[i]
            
            if ((labelMatrix[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMatrix[i]*Ei > toler) and (alphas[i] > 0)):
                #如果误差很大，可以对该数据实例所对应的α值进行优化
                j = selectJrand(i,m) #选择一个α对
                #步骤1:计算这个α所对应的数据的误差
                fXj = float(np.multiply(alphas,labelMatrix).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMatrix[j])
                #保存更新前的内容
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy();
                
                #步骤2:开始计算上下界，保证α在0和C之间
                if (labelMatrix[i] != labelMatrix[j]):
                    #如果ai和aj所对应的数据属于不同的类别
                    L = max(0, alphas[j] - alphas[i]) #下界
                    H = min(C, C + alphas[j] - alphas[i]) #上界
                else:
                    #如果ai和aj所对应的数据属于相同的类别
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                    #如果上下界相同，就不需要调整了，continue跳过
                    print("L==H")
                    continue
                
                #步骤3:计算eta（a[j]）的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                
                if eta >= 0: 
                    print("eta>=0") 
                    continue
                
                #步骤4:更新eta
                alphas[j] -= labelMatrix[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L) #对a[j]进行修剪
                
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                    #a[j]的变化太小了，重新再来
                    print("a[j]的变化太小了，重新再来")
                    continue
                
                #步骤5:更新a[i]
                alphas[i] += labelMatrix[j]*labelMatrix[i]*(alphaJold - alphas[j])#update i by the same amount as j
                
                #更新b1和b2
                b1 = b - Ei- labelMatrix[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMatrix[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMatrix[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMatrix[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                #更新b
                if (0 < alphas[i]) and (C > alphas[i]): 
                    #0 < a[i] < C
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    #0 < a[j] < C
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                
                alphaPairsChanged += 1 #成功更新一次α对        
                print("iter: %d i:%d, α对改变次数: %d" % (iter,i,alphaPairsChanged))
        
        #如果所有向量都没被优化，增加迭代数目，继续下一次循环
        if (alphaPairsChanged == 0): 
            iter += 1
        #要在所有数据集上遍历manIter次
        else: 
            iter = 0
        
        print("当前迭代数为: %d" % iter)
    
    return b,alphas

def showSupportVector(dataArr, labelArr, alphas, b, fig):
    """
    显示支持向量及分隔超平面
    参数：
        dataArr:特征数组
    """
    #首先，绘制支持向量
    m = len(dataArr) #一共m个样本
    sv_feature1_positive = []
    sv_feature2_positive = []
    sv_feature1_negative = []
    sv_feature2_negative = []
    
    for i in range(m):
        if alphas[i] > 0:
            if labelArr[i] == -1:
                sv_feature1_negative.append(dataArr[i][0])
                sv_feature2_negative.append(dataArr[i][1])
            else:
                sv_feature1_positive.append(dataArr[i][0])
                sv_feature2_positive.append(dataArr[i][1])
    
    
    plt.scatter(sv_feature1_positive, sv_feature2_positive, c='yellow', s=50, label='support')
    plt.scatter(sv_feature1_negative, sv_feature2_negative, c='yellow', s=50)
    
    #计算w
    labelMatrix = np.mat(labelArr).T
    dataMatrix = np.mat(dataArr)
    w = (np.multiply(alphas, labelMatrix)).T * dataMatrix
    w = w.tolist()
    w1, w2 = w[0][0], w[0][1]
    
    #绘制分隔超平面
    x = np.linspace(1,8,100)
    y = (-b - w1 * x) / w2
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    
    plt.plot(x, y, label="seperate Line")
    plt.legend()
    plt.ylim(-8,6)
    
if __name__ == "__main__":
    #1. 加载数据集
    filename = "testSet.txt"
    dataArr, labelArr = loadDataSet(filename)
    
    #2. 可视化数据集
    fig = plt.figure("数据可视化")
    showData(dataArr, labelArr, fig)    
    
    #3. 利用简单SMO算法
    b, alphas = simple(dataArr, labelArr, 0.6, 0.001, 40)
    
    #4. 绘制超平面（这里是一根线）
    showSupportVector(dataArr, labelArr, alphas, b, fig)
    
    
