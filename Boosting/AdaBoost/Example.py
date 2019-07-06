# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:14:22 2019
实现统计学习方法书中的AdaBoost例题
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt


def loadSimpleData():
    """创建数据"""
    data = [0,1,2,3,4,5,6,7,8,9]
    dataMat = np.matrix(data).T
    classLabels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    return dataMat, classLabels

def stumpClassify(dataMat, threshVal, threshIneq):
    """
    通过阈值比较对数据进行分类
    参数：
    dataMat:数据矩阵
    threshVal:阈值
    threshIneq:标记
    """
    retArray = np.ones(np.shape(dataMat)) #（样本个数，1）
    if threshIneq == 'lt':#错误率less than
        retArray[dataMat[:, 0] <= threshVal] = -1
    else:#错误率greater than
        retArray[dataMat[:, 0] > threshVal] = -1
    return retArray

def buildStump(dataArr,classLabels,D):
    """
    找到数据集上最佳的单层决策树
    参数：
    dataArr:特征数组
    classLabels:标签数组
    D:样本权重
    返回值：
    bestStump:最佳单层决策树信息
    minError:最小误差
    bestClasEst:最佳的分类结果
    """
    labelMat = np.mat(classLabels).T
    m = np.shape(dataArr)[0] #m是样本数
    numSteps = 10.0 #设定步数
    bestStump = {} #用字典存储最佳单层决策树的相关信息
    bestClasEst = np.mat(np.zeros((m,1))) #初始化分类结果为1
    minError = np.inf #将最小错误率初始化为无穷
    rangeMin = dataArr[:,0].min() #计算该特征的极值
    rangeMax = dataArr[:,0].max()
    stepSize = (rangeMax-rangeMin)/numSteps #计算步长
    for j in range(-1, int(numSteps) + 1):
        #numSteps+2次循环
        for inequal in ['lt', 'gt']: 
            #分别尝试less than和greater than
            threshVal = (rangeMin + float(j) * stepSize) #逐渐增加阈值
            #用第i个特征来划分数据集，阈值为threshVal。符号为inequal
            predictedVals = stumpClassify(dataArr,threshVal,inequal)
            errArr = np.mat(np.ones((m,1))) #错误率向量（m，1）初始化为1
            errArr[predictedVals == labelMat] = 0 #预测正确的，变为0
            weightedError = D.T*errArr  #计算误差
            
            if weightedError < minError:
                #如果误差小于最小误差就将当前误差作为最小误差
                minError = weightedError
                bestClasEst = predictedVals.copy()
                #记录相应的划分信息
                bestStump['thresh'] = threshVal#所用阈值
                bestStump['ineq'] = inequal#标记
    return bestStump, minError, bestClasEst

def adaboostTrainDS(dataArr, classLabels, numIt=40):
    """
    利用adboost训练数据集
    参数：数据集、类标、迭代次数（默认为40）
    输出：分类结果
    """
    weakClassArr = []
    m = np.shape(dataArr)[0] #样本个数
    D = np.mat(np.ones((m,1)) / m) #初始化样本权重矩阵，包含了所有样本的权重
    aggClassEst = np.mat(np.zeros((m,1)))#记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("当前第%d次循环"%i)
        #实际上α值显示了分类器分类的效果，分类越准确，α值越大
        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))#每个分类器的权重
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("当前单层决策树分类结果:", classEst.T)
        #计算下一次迭代的权重向量D，分类正确的权重减小，分类错误的权重增大
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        #计算错误率
        aggClassEst += alpha * classEst
        print("累计值：", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        print("错误率：", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

if __name__ == "__main__":
    dataMat , classLabels = loadSimpleData()
    weakClassArr, aggClassEst = adaboostTrainDS(dataMat, classLabels, 40)
    classify_result = np.sign(aggClassEst)
    print("弱分类器数组：", weakClassArr)
    print("累计分类结果：", aggClassEst)
    print("最终分类结果：", classify_result)
