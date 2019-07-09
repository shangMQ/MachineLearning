# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:57:17 2019
二分K-均值算法的简单实现
采用二分K-均值算法时为了解决KMeans算法收敛于局部最小值的问题
@author: 尚梦琦
"""
import numpy as np
import matplotlib.pyplot as plt
from simple_kmeans import kMeans, distEclud, showClusterData

def loadDataSet(filename):
    """
    加载数据集函数(m行n列，m代表样本数，n代表特征数)
    参数：文件名
    返回值：数据集列表
    """
    dataSet = []
    fr = open(filename)
    
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        floatLine = [float(curLine[i]) for i in range(len(curLine))]
        dataSet.append(floatLine)
    
    fr.close()
    return dataSet

def showData(dataSet):
    """
    可视化原始数据集
    """
    dataArray = np.array(dataSet)
    fig = plt.figure("原始数据集")
    feature1 = dataArray[:, 0]
    feature2 = dataArray[:, 1]
    plt.scatter(feature1, feature2)
    plt.title("Initial Data")
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    
def biKmeans(dataSet, k, distMeas=distEclud):
    """
    二分kmeans的核心实现
    参数：
        dataSet：数据集
        k：簇数
        distMeas：距离采用欧式距离
    返回值：
        簇的质心，样本聚类后的信息矩阵
    """
    dataMat = np.mat(dataSet)
    m = np.shape(dataMat)[0] #计算样本点数
    clusterAssment = np.mat(np.zeros((m,2))) #利用一个(m,2)shape的矩阵存储样本聚类后的情况
    #1. 将所有点看成一个簇
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] #求得训练集中每列元素的均值，作为一个质心
    centList =[centroid0] #放入一个列表中
    
    #2. 计算初始误差
    for j in range(m):
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataMat[j])**2
    
    #3. 只要当前簇数小于目标簇数k，进行循环
    while (len(centList) < k):
        #设置当前最小SSE是np.inf
        lowestSSE = np.inf
        
        #遍历所有的簇，找到最佳划分簇，在上面进行KMeans聚类划分(其中k=2)
        for i in range(len(centList)):
            ptsInCurrCluster = dataMat[np.nonzero(clusterAssment[:,0].A==i)[0],:]#获得当前属于聚簇类的样本元素
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)#对其进行二分类处理
            sseSplit = np.sum(splitClustAss[:,1])#计算划分后的SSE总和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])#计算剩余样本的误差和
            
            #划分误差和剩余误差和作为本次划分的误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                #如果比最小SSE小，就记录相应的划分信息
                bestCentToSplit = i #第i个簇
                bestNewCents = centroidMat #二分之后的簇的质心值
                bestClustAss = splitClustAss.copy() #将对应的二分矩阵保存起来
                lowestSSE = sseSplit + sseNotSplit  #将当前最小误差设置为划分误差和剩余误差的总和
        
        #选择使得误差最小的那个簇进行划分操作，划分最佳簇后会得到两个索引为0和1的子簇
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #将索引为1的子簇划分出去
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit #索引为0的子簇保持原来的簇索引
        print('此次选择的最佳划分簇索引是: ',bestCentToSplit)
        print('该簇的元素个数为：', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#替换划分后的第一个簇的质心值
        centList.append(bestNewCents[1,:].tolist()[0]) #添加新簇
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return np.mat(centList), clusterAssment

if __name__ == "__main__":
    dataSet = loadDataSet("testSet2.txt")
    showData(dataSet)
    dataMat = np.mat(dataSet)
    k = 3
    centorids, clusterAssment = biKmeans(dataSet, k)
    showClusterData(dataMat, centorids, clusterAssment, k)
