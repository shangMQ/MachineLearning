# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:10:44 2019
AdaBoost算法在马疝病数据集上的应用
@author: Kylin
"""
import numpy as np
from adaboostingTest import adaboostTrainDS, adaClassify

def loadDataSet(filename):
    #自适应数据加载函数，自动检测出特征数目，同时，假定最后一个特征是类别标签
    numFeat = len(open(filename).readline().split('\t'))
    data = []
    labels = []
    fr = open(filename)
    
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        data.append(lineArr)
        labels.append(float(curLine[-1]))
    
    return data, labels

if __name__ == "__main__":
    trainfile = "horseColicTraining2.txt"
    testfile = "horseColicTest2.txt"
    #加载数据集
    data, labels = loadDataSet(trainfile)
    testdata, testlabels = loadDataSet(testfile)
    trainlabelsmat = np.mat(labels)
    testlabelsmat = np.mat(testlabels)
    train_m = np.shape(data)[0]
    test_m = np.shape(testdata)[0]
    
    #设置最大弱分类器个数
    IterList = [1, 10, 50, 100]
    trainerrorList = []
    testerrorList = []
    
    for i in IterList:
        #分别将最大若分类器个数修改为列表中的值
        #在训练集上得到决策树数组
        classifierArray, aggClassEst = adaboostTrainDS(data, labels, i)
        
        #查看训练集上的效果
        train_prediction = adaClassify(data, classifierArray)
        trainErrArr = np.mat(np.ones((train_m,1)))
        trainerror = trainErrArr[train_prediction != trainlabelsmat.T].sum()
        trainerrRatio = trainerror / train_m
        trainerrorList.append(trainerrRatio)
        
        #查看测试集上的效果
        prediction = adaClassify(testdata, classifierArray)
        testErrArr = np.mat(np.ones((test_m,1)))
        testerror = testErrArr[prediction != testlabelsmat.T].sum()
        testerrRatio = testerror / test_m
        testerrorList.append(testerrRatio)
        
    print("训练集上的错误率：", trainerrorList)
    print("测试集上的错误率：", testerrorList)
    #可以发现最大分类器达到50次之后，就会产生过拟合问题
    