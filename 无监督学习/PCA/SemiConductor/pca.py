# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:55:22 2019
利用PCA对半导体数据进行降维
@author: Kylin
"""
import numpy as np

def loadData(filename, delim="\t"):
    """
    加载数据集
    参数：
        filename:文件名
        delim:分隔符，默认为制表符"\t"
    返回值:
        dataMat:数据矩阵
    """
    f = open(filename, 'r')
    data = []
    #此时获取的数据均为字符串格式,需要将其转换为float格式
    for line in f.readlines():
        numstring = line.strip().split(delim)
        subdata = []
        for num in map(float, numstring):
            subdata.append(num)
        data.append(subdata)
    
    #将数据转换为矩阵形式
    dataMat = np.mat(data)
    return dataMat

def replaceNanWithMean():
    """
    利用均值替换为NAN的数据
    """
    dataMat = loadData("secom.data", " ")
    FeatureNum = dataMat.shape[1]
    print(FeatureNum)
    

