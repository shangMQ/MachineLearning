# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:59:46 2019
朴素贝叶斯分类器
@author: Kylin
"""
import numpy as np
import pandas as pd

def loadDataSet():
    """
    加载数据集，创建了一些实验样本
    返回值：进行词条划分后的文档集合和类标集合
    """
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1表示禁语，0表示正常言论
    return postingList,classVec

def createVocabList(dataSet):
    """
    创建词汇表，去除了文档中的重复词汇
    返回值：词汇表
    """
    vocabSet = set([])  #创建空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取两个集合的并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    输入参数：词汇表及某个文档
    输出：值为0或1的文档向量，表示词汇表中的单词在文档中是否出现过
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word \"{:}\" bot in VocabList".format(word))
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    """
    计算属于侮辱性文档的概率以及两个类别的概率向量
    输入：训练集矩阵（01元素构成），类别向量
    输出：非侮辱性句子各个词的概率向量，侮辱性句子各个词的概率向量，文档属于侮辱性文档的概率
    """
    numTrainDocs = len(trainMatrix) #文档个数（6个）
    numWords = len(trainMatrix[0]) #单个训练集中的词汇数（32个）
    pAbusive = sum(trainCategory)/float(numTrainDocs)#计算文档属于侮辱性文档的概率pc1
    p0Num = np.ones(numWords) #计算p(wi|c1)或p(wi|c0)
    p1Num = np.ones(numWords)   
    p0Denom = 2.0
    p1Denom = 2.0  #change to 2.0初始化分母

    for i in range(numTrainDocs):
        if trainCategory[i] == 1: 
            #如果第i句是侮辱句，则p1分子+=相应值，分母+=为1的单词总和
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            #如果第i句不是侮辱句，则p0做相同操作
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #分类别计算每个词出现的概率
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)      
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    计算是否为侮辱性句子
    输入：待分类的向量，p0Vec, p1Vec, 句子为侮辱性句子的概率
    输出：0非侮辱性句子，1侮辱性句子
    """
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        #返回概率大的类别
        return 1
    else: 
        return 0

def testingNB():
    """
    测试朴素贝叶斯算法实现文档分类
    """
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['I', 'love', 'Kylin']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['YSL', 'is', 'stupid']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


if __name__ == "__main__":
    testingNB()
    
    