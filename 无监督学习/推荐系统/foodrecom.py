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
    """
    n = np.shape(dataMat)[1] #获取菜品种类个数
    simTotal = 0.0 #初始化两个变量值
    ratSimTotal = 0.0
    
    for j in range(n):
        #遍历n个菜品
        userRating = dataMat[user, j] #获取相应用户对某个菜品的评分
        if userRating == 0:
            #用户未对该菜品进行评价
            continue
        overLap = np.nonzero()
        
    
    
    

