# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:41:42 2019
餐馆菜品推荐系统
@author: Kylin
"""

import numpy as np


def standEst(dataMat, user, similarMeasurment, item):
    """
    计算在给定相似度计算方法下，用户对物品的估计评分值
    参数：
    dataMat：数据矩阵
    user：用户编号
    similarMeasurment：相似度计算方法
    item：物品编号
    返回值：最终评分
    """
    n = np.shape(dataMat)[1] #获取菜品种类个数
    simTotal = 0.0 #初始化两个变量值
    ratSimTotal = 0.0
    
    for j in range(n):
        #遍历n个菜品，j是每一个菜品
        userRating = dataMat[user, j] #获取相应用户对某个菜品的评分
        if userRating == 0:
            #用户未对该菜品进行评价，直接跳过
            continue
        #寻找两个用户都评级的物品
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A > 0,
                             dataMat[:,j].A > 0))[0]
        
        if len(overLap) == 0:
            #如果两者没有任何重合的元素，则相似度为0
            similarity = 0
        else:
            #否则基于这两个物品计算相似度
            similarity = similarMeasurment(dataMat[overLap[:,item], dataMat[:,j]])
        print("{:d}和{:d}的相似度是{:.2f}".format(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
    
def recommend(dataMat, user, N = 3, similarMeas, estMethod=standEst):
    """
    产生最高的N个推荐结果
    参数：
    dataMat:数据矩阵
    user:用户编号
    N:需要产生的推荐结果数量，默认为3
    similarMeas:相似度计算方法
    estMethod:评估所采用的方法
    """
    #首先，获取未评估菜品的个数
    unratedItems = np.nonzero(dataMat[user,:].A == 0)[0]
    if len(unratedItems) == 0:
        print("所有菜品均已评级")
    itemScores = []
    for item in unratedItems:
        #对每个未评级菜品估计评分
        estimatedScore = estMethod(dataMat, user, similarMeas, item)
        itemScores.append((item,estimatedScore))
        
 
