# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:23:09 2019
梯度上升法
@author: Kylin
"""
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet():
    """
    读取文件testSet.txt文件中的内容，
    每行前两个值为X1和X2， 第三个值是数据对应的类别标签。
    为了方便计算，将X0设置为1。
    返回值：数据集dataMat，标签labelMat
    """
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
        
    fr.close()
    return dataMat,labelMat

def sigmoid(inX):
    #计算sigmoid函数
    return 1.0 / (1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
    参数：二维dataMatIn列表，以及标签列表classLabels
    与感知器模型的思想一致
    输出：最佳参数
    """
    dataMatrix = np.mat(dataMatIn)#转化为numpy矩阵matrix
    labelMat = np.mat(classLabels).transpose() #转化我numpy矩阵matrix并转置
    m,n = np.shape(dataMatrix) #m行n列（100行3列）
    alpha = 0.001#向目标移动的步长
    maxCycles = 500 #最大迭代次数
    weights = np.ones((n,1)) #将最佳回归系数的初始值设置为1
    for k in range(maxCycles):#注意：这里的运算为矩阵运算
        h = sigmoid(dataMatrix*weights) #结果为(100,1)的矩阵
        error = (labelMat - h) #与真实的类标之间的差距
        #按照差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose()* error 
    return weights

def plotBestFit(weights):
    """
    画出数据集和Logistic回归最佳拟合直线的函数
    """
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0] #数据集有100行元素 
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            #如果是1分类中的
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')#1类
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    #X2与X1的关系式
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升算法
    """
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n) #将回归系数初始化为1
    data = np.array(dataMatrix)
    for i in range(m): 
        """
        计算每一个样本的梯度
        """
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights +=  alpha * error * data[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升算法
    """
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    data = np.array(dataMatrix)
    
    for j in range(numIter): 
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001 #α在每次迭代的时候都会调整，缓解数据波动，但是不会减少到0.
            #随机选取更新，randIndex是一个随机的均匀分布数,减少周期性的波动
            randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(data[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * data[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    """
    参数：特征向量，回归系数
    返回类别1或者0
    """
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: 
        return 1.0
    else: 
        return 0.0

def colicTest():
    #准备数据
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    frTrain.close()
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    frTest.close()
    errorRate = (float(errorCount)/numTestVec)
    print("测试集的错误率是: ",errorRate)
    return errorRate

def multiTest():
    """
    调用colicTest()10次并求值
    """
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    errormean = errorSum/float(numTests)
    print("{:}次测试的平均错误率为{:.2f}".format(numTests, errormean))
        

if __name__ == '__main__':
    multiTest()

    

