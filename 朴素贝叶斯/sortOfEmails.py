# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:59:35 2019
利用朴素贝叶斯实现邮件分类
@author: Kylin
"""
import re
import naiveBayes
import random
import numpy as np

def textParse(bigString): 
    """
    文件解析函数，
    输入：邮件文本
    输出：词汇集合，过滤掉小于3的词汇
    """
    listOfTokens = re.split(r'\W*', bigString)
    tokens = [token.lower() for token in listOfTokens if len(token) > 2]
    return tokens 
    
def spamTest():
    """
    使用朴素贝叶斯算法对垃圾邮件自动分类
    """
    docList = [] #文档列表
    classList = [] #文档类别列表
    fullText = []
    #1.解析并导入文本文件
    for i in range(1,26):
        #先处理标记为spam（垃圾邮件）的文件
        spamfilename = 'email/spam/%d.txt' % i
        spamfile = open(spamfilename, 'r')
        text = spamfile.read()
        wordList = textParse(text)
        spamfile.close()
        docList.append(wordList)#将词汇加入文档列表中
        fullText.extend(wordList)#将词汇列表扩充进fullText列表中
        classList.append(1) #加入类标为1
        
        #处理正常邮件
        hamfilename = 'email/ham/%d.txt' % i
        hamtext = open(hamfilename, 'r').read()
        wordList = textParse(hamtext)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #加入类标为0
    vocabList = naiveBayes.createVocabList(docList)#创建无重复的词汇表
    
    #2. 随机构建训练集和测试集
    trainingSet = list(range(50)) #已知一共有50个邮件样本 
    testSet=[]  #创建测试集
    for i in range(10):
        #随机选取10个作为测试集样本，并将其从训练集中剔除
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    
    #3. 构建向量词矩阵    
    trainMat=[] #01元素构成的训练集矩阵
    trainClasses = []#文档类别列表
    for docIndex in trainingSet:
        #遍历训练集，构建向量词矩阵
        #将词包加入训练集矩阵
        trainMat.append(naiveBayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])#将相应的类别加入类别列表
    
    #4. 计算分类的向量，以及分类为垃圾邮件的概率
    p0V,p1V,pSpam = naiveBayes.trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0 #初始化错误分类
    for docIndex in testSet:        
        #遍历测试集，对测试集进行分类
        wordVector = naiveBayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if naiveBayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("当前分类错误，第{:}个邮件分类错误".format(docIndex))
    print('错误率为: ', errorCount/len(testSet))
    
    
if __name__ == '__main__':
    spamTest()



