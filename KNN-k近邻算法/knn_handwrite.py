# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:23:59 2019
手写识别系统
@author: Kylin
"""

import numpy as np
from os import listdir
from knntest import classify


#编写函数img2vector()，将图像格式转化为分类器使用的向量格式

def img2vector(filename):
    #将图像转换为向量
    '''必须将图像格式化处理为一个向量，我们把32×32的二进制图像矩阵转换为1×1024的向量，
    这样就可以使用classifier分类器了。
    '''
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#4. 测试算法
def handwritingClassTest():
    #1. 收集数据:提供文本文件
    #trainingDigits中包含了2000个例子，testDigits中包含了大学900个测试数据
    hwLabels = []
    trainingFileList = listdir('trainingDigits') #加载训练集
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i] #获得第i个文件名
        fileStr = fileNameStr.split('.')[0] #获得不含后缀的文件名
        #classNumStr为类标
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #2. 将数据转化为分类器所需要的格式
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
   
    #3. 加载测试集
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0] #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #4. 训练数据
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"%(classifierResult, classNumStr))
        if (classifierResult != classNumStr): 
            errorCount += 1.0
    #5. 输出测试结果，及总体的错误率
    print("\n在测试集中的错误次数为: %d" % errorCount)
    print("\n总体错误率是: %f" % (errorCount/float(mTest)))

if __name__ == '__main__':
    handwritingClassTest()
