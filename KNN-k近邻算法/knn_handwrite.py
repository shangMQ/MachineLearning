# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:23:59 2019
手写识别系统
@author: Kylin
"""

import numpy as np
from os import listdir


def img2vector(filename):
    """
    将图像转换为向量
    必须将图像格式化处理为一个向量，我们把32×32的二进制图像矩阵转换为1×1024的向量，
    这样就可以使用classifier分类器了。
    :param filename:
    :return:
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def classify(inX, dataSet, labels, k):
    """
        参数简介：
        inX: 用于分类的输入向量
        dataSet: 输入的训练样本集
        labels: 标签向量
        k: 选择最近邻居的数目
    """
    dataSetSize = dataSet.shape[0]

    # 1. 计算已知类别数据集中的点与当前点之间的距离
    # np.tile()将用于分类的输入向量纵向赋值为dataSetSize个, 在与每个dataSet作差
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    # 为了避免出现负数，平方，因为距离均为正值
    sqDiffMat = diffMat ** 2
    # 计算和(按照行求和)
    sqDistances = sqDiffMat.sum(axis=1)
    # 再开方，计算出与每个训练集中的欧式距离
    distances = sqDistances ** 0.5

    # 2. 按照距离递增顺序排序，返回一个下标数组
    sortedDistIndicies = distances.argsort()

    # 3. 选取与当前点距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 4. 确定前k个点所在类别出现的频率,operator.itemgetter()用于获取对象哪些维的数据
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)

    # 5. 返回前k个点出现频率最高的类别作为当前点的预测类别。
    return sortedClassCount[0][0]


def handwritingClassTest():
    """
    手写数字识别算法
    :return:
    """
    # 1. 收集数据:提供文本文件
    # trainingDigits中包含了2000个例子，testDigits中包含了大学900个测试数据
    hwLabels = []
    trainingFileList = listdir('trainingDigits') # 加载训练集
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 获得第i个文件名
        fileStr = fileNameStr.split('.')[0]  # 获得不含后缀的文件名
        
        # classNumStr为类标
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 2. 将数据转化为分类器所需要的格式
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
   
    # 3. 加载测试集
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0] #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        
        # 4. 训练数据
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): 
            errorCount += 1.0
            
    # 5. 输出测试结果，及总体的错误率
    print("\n在测试集中的错误次数为: %d" % errorCount)
    print("\n总体错误率是: %f" % (errorCount/float(mTest)))

if __name__ == '__main__':
    handwritingClassTest()
