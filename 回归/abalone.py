# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:24:38 2019
预测鲍鱼的年龄
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt
from OLSregression import loadDataSet
from localweightLinearRegression import lwlrTest

def rssError(yArr, yHat):
    return ((yArr - yHat) ** 2).sum()

#获得鲍鱼的样本数据，abX是特征数组，abY是真实值数组
abX, abY = loadDataSet('abalone.txt')
k = [0.1, 1, 10]
regressError = []
for i in range(len(k)):
    yHat = lwlrTest(abX[0:99], abX[0:99], abY[0:99], k[i])
    regressError.append(rssError(abY[0:99], yHat))
    print("当k = {:}时，rss的值为{:.4f}".format(k[i], regressError[i]))
