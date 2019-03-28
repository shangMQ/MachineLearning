# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:35:06 2019
在约会网站上使用k近邻算法
----来自《Machine Learning in Action》
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt
from knntest import classify

def file2matrix(filename):
    '''
         处理输入格式问题函数，输入为文件名，输出为训练样本矩阵和类标签向量   
    '''
    #准备数据，使用Python解析文本文件
    fr = open(filename)
    numberOfLines = len(fr.readlines())  #得到文件中的行数
    returnMat = np.zeros((numberOfLines,3))  #准备训练矩阵
    classLabelVector = []   #准备标签向量   
    fr = open(filename) #重新加载一次，因为默认readlines会读到末尾
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


#归一化数据
'''
    处理不同取值范围的特征值时，通常采用将数值归一化，例如将取值范围处理为0到1或者-1到1之间。
'''
def autoNorm(dataSet):
    '''
        自动将数字特征值转化为0到1之间
    '''
    #dataSet.min(0)中的参数0使得函数可以从列中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals
    
    
#1. 收集数据,放在文件datingTestSet2.txt,每个样本数据占一行共3个特征，共1000行
file = "datingTestSet2.txt"
datingDataMat, datingLabels = file2matrix(file)


#2. 分析数据，利用matplotlib绘制散点图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 
           15*np.array(datingLabels), 15*np.array(datingLabels))
plt.show()

#3. 测试算法
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #测试集numTestVecs是样本总数m的10%
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #numTestVecs利用normMat的前i行数据进行测试，之后的用于训练集，k=3
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

#5. 使用算法
def classifyPerson():
    resultList = ['不喜欢', '一般', '很喜欢']
    print("-"*5, "Kylin家园交友网站", "-"*5)
    print("请输入男嘉宾的相关信息:")
    percentTats = float(input("玩游戏所消耗的时间百分比？"))
    ffMiles = float(input("每年的飞行里程有多少？"))
    iceCream = float(input("每周消耗的冰激凌公升数？"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("你对他可能：",resultList[classifierResult-1])

if __name__ == '__main__':
    classifyPerson()