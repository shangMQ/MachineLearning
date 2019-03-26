# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:59:42 2019
K近邻算法练习
'''
    使用k近邻算法将每组数据划分到某个类中
'''
@author: Kylin
"""
import numpy as np
import operator #导入运算符模块

#创建带标签的数据集
def createDataSet():
    group = np.array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    '''
        参数简介：
        inX: 用于分类的输入向量
        dataSet: 输入的训练样本集
        labels: 标签向量
        k: 选择最近邻居的数目
    '''
   
    dataSetSize = dataSet.shape[0]
    
    #1. 计算已知类别数据集中的点与当前点之间的距离
    
    #np.tile()将用于分类的输入向量纵向赋值为dataSetSize个, 在与每个dataSet作差
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    #为了避免出现负数，平方，因为距离均为正值
    sqDiffMat = diffMat**2
    #计算和(按照行求和)
    sqDistances = sqDiffMat.sum(axis=1)
    #再开方，计算出与每个训练集中的欧式距离
    distances = sqDistances**0.5
    
    #2. 按照距离递增顺序排序，返回一个下标数组
    sortedDistIndicies = distances.argsort()    
    
    #3. 选取与当前点距离最小的k个点
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    
    #4. 确定前k个点所在类别出现的频率,operator.itemgetter()用于获取对象哪些维的数据
    sortedClassCount = sorted(classCount.items(), 
                              key=operator.itemgetter(1), reverse=True)
    
    #5. 返回前k个点出现频率最高的类别作为当前点的预测类别。
    return sortedClassCount[0][0]


#当文件中没有引入其他模块时，返回__main__；当引入到其他模块时，会返回模块名    
if __name__ == '__main__':
   dataset, labels = createDataSet()
   label = classify([0,0], dataset, labels, 3) 
   print("Predict label:{}".format(label))
    



    


