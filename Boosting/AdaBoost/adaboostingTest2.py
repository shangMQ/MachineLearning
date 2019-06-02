# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:41:05 2019
AdaBoost的简单实现
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt

def loadSimpleData():
    """创建数据"""
    data = [[1., 2.1],
            [1.5, 1.6],
            [1.3, 1.],
            [1., 1.],
            [2., 1.]]
    dataMat = np.matrix(data)
    classLabels = [1, 1, -1, -1, 1]
    return dataMat, classLabels

def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    """
    通过阈值比较对数据进行分类
    参数：
    dataMat:数据矩阵
    dimen:第几个特征
    threshVal:阈值
    threshIneq:标记
    """
    retArray = np.ones((np.shape(dataMat)[0],1)) #（样本个数，1）
    if threshIneq == 'lt':#错误率less than
        retArray[dataMat[:, dimen] <= threshVal] = -1
    else:#错误率greater than
        retArray[dataMat[:, dimen] > threshVal] = -1
    return retArray

def showPlot(data, labels):
    label = np.mat(labels).T
    plt.rcParams['font.sans-serif']=['simhei'] #设置显示中文title的字体
    x = np.array(dataMat[:,0])
    y = np.array(dataMat[:,1])
    label = np.array(label)
    plt.scatter(x, y, c=label)
    plt.title("数据可视化")
    plt.show()
    
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
    dataMatrix = np.mat(dataArr);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix) #m是样本数，n是特征数
    numSteps = 10.0 #设定步数
    bestStump = {} #用字典存储最佳单层决策树的相关信息
    bestClasEst = np.mat(np.zeros((m,1))) #初始化分类结果为1
    minError = np.inf #将最小错误率初始化为无穷
    
    for i in range(n):#第一层循环遍历n个特征
        rangeMin = dataMatrix[:,i].min() #计算该特征的极值
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps #计算步长
        for j in range(-1, int(numSteps) + 1):
            #numSteps+2次循环
            for inequal in ['lt', 'gt']: 
                #分别尝试less than和greater than
                threshVal = (rangeMin + float(j) * stepSize) #逐渐增加阈值
                #用第i个特征来划分数据集，阈值为threshVal。符号为inequal
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1))) #错误率向量（m，1）初始化为1
                errArr[predictedVals == labelMat] = 0 #预测正确的，变为0
                weightedError = D.T*errArr  #计算误差
                
                if weightedError < minError:
                    #如果误差小于最小误差就将当前误差作为最小误差
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    #记录相应的划分信息
                    bestStump['dim'] = i#记录第dim个特征
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

def adaClassify(datToClass, classifierArr):
    """
    利用训练出来的多个弱分类器进行分类的函数
    参数：多个待分类样本数组，多个弱分类器组成的数组
    输出：分类结果
    """
    dataMatrix = np.mat(datToClass) #将特征数组转化为特征矩阵
    m = np.shape(dataMatrix)[0] #得到样本数
    aggClassEst = np.mat(np.zeros((m,1))) #初始化累计分类矩阵
    
    for i in range(len(classifierArr)):
        #循环遍历若干个分类器
        print("当前遍历第%d个分类器"%i)
        dim = classifierArr[i]['dim'] #第dim个特征
        thresh = classifierArr[i]['thresh'] #阈值
        ineq = classifierArr[i]['ineq'] #标记
        classEst = stumpClassify(dataMatrix, dim, thresh, ineq)
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print("累计分类值：", aggClassEst)
    result = np.sign(aggClassEst)
    return result

if __name__ == "__main__":    
    #获取数据
    dataMat, classLabels = loadSimpleData()
    print("特征矩阵：")
    print(dataMat)
    print("类标：")
    print(classLabels)
    #可视化数据
    showPlot(dataMat, classLabels)
    #利用AdaBoost对数据进行分类
    weakClassArr, aggClassEst = adaboostTrainDS(dataMat, classLabels, 40)
    print("弱分类器数组：", weakClassArr)
    print("累计分类结果：", aggClassEst)
    preClass = np.sign(aggClassEst)
    print("训练集分类结果：", preClass)
    #在测试集上查看效果
    testArray = [1,2]
    testResult = adaClassify(testArray, weakClassArr)
    print("测试数据上的分类结果:", testResult)