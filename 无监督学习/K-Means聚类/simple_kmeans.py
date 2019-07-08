# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:04:07 2019
简单K-means聚类算法的实现
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    """
    加载数据集函数(m行n列，m代表样本数，n代表特征数)
    参数：文件名
    返回值：数据集列表
    """
    dataMat = []
    fr = open(filename)
    
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        floatLine = [float(curLine[i]) for i in range(len(curLine))]
        dataMat.append(floatLine)
    
    fr.close()
    return dataMat

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

def showClusterData(dataMat, clusterAssignment, k):
    """
    可视化聚簇之后的数据集
    参数：
        dataMat：数据集矩阵
        clusterAssignment：数据点分簇结果
        k: 簇数
    """
    cluster = clusterAssignment.A[:, 0]
    data = dataMat.A
    colors = ['r', 'y', 'b', 'g']
    fig = plt.figure("聚簇后的数据点")
    for i in range(len(dataMat)):
        ki = int(cluster[i])
        plt.scatter(data[i][0], data[i][1], c = colors[ki])
    
    plt.title("After Cluster")
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
        

def distEclud(vecA, vecB):
    """
    计算两个向量的欧式距离
    参数：向量A，向量B
    返回值：A与B之间的欧式距离
    """
    result = np.sqrt(np.sum(np.power(vecA-vecB, 2)))
    return result

def randCent(dataMat, k):
    """
    为给定数据集构建一个包含k个随机质心的集合
    """
    n = np.shape(dataMat)[1] #特征数
    centroids = np.mat(np.zeros((k,n)))
    
    for j in range(n):
        minJ = min(dataMat[:,j])#计算最小值
        rangeJ = float(max(dataMat[:, j]) - minJ)#计算极差
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    
    return centroids

def kMeans(dataMat, k, distMeans = distEclud, createCent = randCent):
    """
    kMeans算法核心实现
    参数：
        dataMat：数据集矩阵
        k: 簇数
        distMeans：用来计算距离的函数，默认使用欧式距离
        createCent：创建初始质心的函数
    返回：
        k个聚簇质心，以及每个样本点的分簇结果
    """
    m, n = np.shape(dataMat) # 样本数
    clusterAssignment = np.mat(np.zeros((m,2)))
    #包含两列，一列记录簇索引，第二列来存储误差
    
    #1， 确定k个初始点作为质心（随机选择k个样本）
    centroids = createCent(dataMat, k)
    
    clusterChanged = True#循环标志变量
    
    #2. 当任意一个点的簇分配结果发生改变
    while clusterChanged:
        clusterChanged = False
        
        #对数据集中的每个数据点
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            
            #对每个质心
            for j in range(k):
                #计算质心与数据点中的距离
                distJI = distMeans(centroids[j], dataMat[i])
                
                #寻找最近的质心
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            
            #将数据点分配到距离其最近的点
            if clusterAssignment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssignment[i, :] = minIndex, minDist**2
        print("聚簇：", centroids)    
        
        #对每一个簇，计算簇中所有点的均值并将均值作为质心
        for cent in range(k):
            ptsInClust = dataMat[np.nonzero(clusterAssignment[:,0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)#axis=0表示按列方向进行均值计算
        
    return centroids, clusterAssignment


if __name__ == "__main__":
    dataSet = loadDataSet("testSet.txt")
    showData(dataSet)
    dataMat = np.mat(dataSet)
    print("最小值:", min(dataMat[:,0]), min(dataMat[:,1]))
    print("最大值:", max(dataMat[:,0]),max(dataMat[:,1]))
    k = int(input("请输入k值:"))
    centroids, clusterAssignment = kMeans(dataMat, k)
    showClusterData(dataMat, clusterAssignment, k)
    
    
    