# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:37:22 2019

@author: Kylin
"""
import math
import operator

def calcShannonEnt(dataSet):
    """
    计算信息熵的函数，输入参数是数据集，输出是对应数据集的信息熵
    """
    #计算数据集中实例的总数
    numEntries = len(dataSet)
    #创建一个数据字典，它的键是数据集中的最后一列的数值
    labelCounts = {}
    #如果当前键值不存在，则扩展字典并将当前键值加入字典，
    #每个键值都记录了当前类别出现的次数。
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #信息熵求和变量
    shannonEnt = 0.0
    #概率用频率来计算
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2) #log base 2
    #最后返回与输入信息集对应的信息熵
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """按照给定的轴划分数据集，
    参数：数据集、待划分数据集的特征和需要返回的特征值
    返回值为满足给定axis的value值的数据列表
    """
    retDataSet = [] #为了不修改原始数据集，重新创建一个列表
    for featVec in dataSet: 
        #循环遍历整个数据集，一旦发现符合要求的值，将其加入retDataSet中
        #当按照某个特征划分数据集时，就需要将所有符合要求的元素抽取出来
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    实现选取特征，划分数据集，计算得出最好的划分数据集的特征
    """
    numFeatures = len(dataSet[0]) - 1 #最后一列是类标，不是特征元素
    baseEntropy = calcShannonEnt(dataSet) #计算整个数据集的原始香农熵
    bestInfoGain = 0.0 #初始化条件熵为0
    bestFeature = -1
    for i in range(numFeatures): #遍历所有特征
        featList = [example[i] for example in dataSet] #将第i个特征的每个数据的值放入featlist列表
        uniqueVals = set(featList) #转化为元素不同的集合
        newEntropy = 0.0
        for value in uniqueVals: #计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        #信息增益=原始信息熵-条件熵总和
        infoGain = baseEntropy - newEntropy  #计算信息增益，熵的减小
        if (infoGain > bestInfoGain):  #比较所有特征中的信息增益
            bestInfoGain = infoGain #如果当前熵大于最佳熵，则将当前的划分方式最为最佳划分方式
            bestFeature = i #将特征号存储在bestFeature中
    return bestFeature    #返回最佳划分特征号

def majorityCnt(classList):
    """
    如果数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时需要决定如何定义该叶子节点，采取多数表决的方式。
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount[vote] += 1
    #将classCount按值从大到小排序
    sortedClassCount = sorted(classCount.items(), 
                              key=operator.itemgetter(1), 
                              reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    """
    创建树的函数
    输入：数据集，标签
    输出：树
    """
    #classList中存储着数据集中的所有类标
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        #如果所有的标签都完全相同，表明数据集中的样本属于同一类，
        #则返回该类标
        return classList[0]
    if len(dataSet[0]) == 1:
        #没有多余的属性用于划分，则返回出现次数最多的类标
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)#选择最优划分属性下标
    bestFeatLabel = labels[bestFeat]#获得相应的类标
    myTree = {bestFeatLabel:{}}#以相应的类标为键，创建一个myTree字典
    del(labels[bestFeat]) #在类标中删除该类标
    #找到数据集中各个数据在该属性上的值
    featValues = [example[bestFeat] for example in dataSet] 
    uniqueVals = set(featValues)#转化为无重复数据的集合
    for value in uniqueVals:
        """
        遍历bestFeat属性的各个值
        """
        subLabels = labels[:] #赋值类标，以防修改labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                  bestFeat, value),
                                                  subLabels)
    return myTree               

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree, fw, 0)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    myDat, labels = createDataSet()
    newlabels = labels.copy()
    Tree = createTree(myDat, newlabels)
    storeTree(Tree, "classifierStorage.txt")
    print(grabTree("classifierStorage.txt"))
    
    
