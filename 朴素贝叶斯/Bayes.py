# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:40:46 2019
统计学习方法-朴素贝叶估计-贝叶斯估计法例题的实现
@author: Kylin
"""
import numpy as np

def changemode(datalist):
    """
    为了便于处理数据，修改data中的第二列元素
    参数：datalist数据列表
    返回值：修改后的数据列表
    """
    alphabat = {'S':0, 'M':1, 'L':2}
    for sonlist in datalist:
        sonlist[1] = alphabat[sonlist[1]]
    return datalist

def loadDataSet():
    """
    加载数据集,数据集有两个特征
    返回值：特征矩阵，类别数组
    """
    datalist = [[1, 'S'],
                [1, 'M'],
                [1, 'M'],
                [1, 'S'],
                [1, 'S'],
                [2, 'S'],
                [2, 'M'],
                [2, 'M'],
                [2, 'L'],
                [2, 'L'],
                [3, 'L'],
                [3, 'M'],
                [3, 'M'],
                [3, 'L'],
                [3, 'L']]
    
    labels = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]).ravel()
    
    #为了便于处理，将第二个特征的字母也映射为数字
    changemode(datalist)
    data = np.array(datalist)
    return data, labels
            
def Bayes(data, labels, newsample, lamda):
    """
    朴素贝叶斯算法
    参数：
        data:特征数组
        labels:类别数组
        newsample:新样本数组
    返回值：
        新样本类别预测结果
    """
    #1. 计算先验概率yk
    y = np.zeros((1,2)).ravel()
    n = len(data)
    for label in labels:
        if label == -1:
            y[0] += 1
        else:
            y[1] += 1
    yk = (y + 1*lamda) / (n + 2 * lamda)
    print("先验概率为：", yk)
    
    #2. 计算条件概率
    #计算特征的可能取值个数
    feature1 = list(set(data[:,0].tolist()))
    feature2 = list(set(data[:,1].tolist()))
    #因为set集合属于无序的，这里要重新排下序
    feature1.sort()
    feature2.sort()
    print("Feature 1:", feature1)
    print("Feature 2:", feature2)
    num1 = len(feature1)
    num2 = len(feature2)
    
    k = np.zeros(((num1 + num2),2))
    k += lamda
    
    for j in range(num1):
        for i, sample in enumerate(data):
            if sample[0] == feature1[j]:
                if labels[i] == -1:
                    k[j,0] += 1
                else:
                    k[j,1] += 1
    for w in range(num1, num1+num2):
        for i, sample in enumerate(data):
            if sample[1] == feature2[w-num1]:
                if labels[i] == -1:
                    k[w,0] += 1
                else:
                    k[w,1] += 1
    #相除，计算条件概率概率
    k[:,0] /= (y[0] + num1 * lamda)
    k[:,1] /= (y[1] + num2 * lamda)
    print("每种情况下的条件概率：")
    print(k)
    
    #3. 预测新样本的类别
    item1 = feature1.index(newsample[0])
    item2 = feature2.index(newsample[1])
    
    result0 = yk[0] * k[item1,0] * k[item2+num1,0]
    result1 = yk[1] * k[item1,1] * k[item2+num1,1]
    print("新样本的预测概率计算结果：", result0, result1)
    if result0 > result1:
        return -1
    else:
        return 1

if __name__ == "__main__":
    #1. 加载数据集
    data, labels = loadDataSet()
    #2. 输入新样本
    sample = [[2, 'S']]
    sample = np.array(changemode(sample)).ravel()
    #3. 预测新样本类型
    result = Bayes(data, labels, sample, 1)
    print("新样本类别预测为：", result)
