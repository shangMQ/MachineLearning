# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:53:25 2019
#绘制ROC曲线
@author: Kylin
"""

import matplotlib.pyplot as plt
import numpy as np
from adaboostingTest import adaboostTrainDS
from AdaHorseColic import loadDataSet

def plotROC(predStrengths, classLabels):
    """
    参数：分类器的预测强度predStrengths，类标classLabels
    """
    cur = (1.0, 1.0)#绘制光标的位置
    ySum = 0.0 #计算AUC值
    numPosClas = sum(np.array(classLabels) == 1.0)#统计正例个数
    yStep = 1 / float(numPosClas) #y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas) #x轴步长
    sortedIndicies = predStrengths.argsort()
    
    plt.rcParams['font.sans-serif']=['simhei'] #设置显示中文title的字体
    fig = plt.figure("ROC曲线")
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1],[0,1], "b--", label="random selection")
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.title("利用AdaBoost预测马疝病的ROC曲线")
    ax.axis([0, 1, 0, 1])
    plt.legend()
    plt.show()
    print("AUC值为：", ySum * xStep)        
            
if __name__ == "__main__":
    #加载数据
    datArr, labelArr = loadDataSet("horseColicTraining2.txt")
    classifierArray, aggClassEst = adaboostTrainDS(datArr, labelArr, 50)
    plotROC(aggClassEst.T, labelArr)
    
    