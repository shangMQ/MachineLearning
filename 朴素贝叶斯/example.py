# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:50:44 2019
简单实现《统计学习方法》中朴素贝叶斯这一节的例题
@author: 尚梦琦
"""
import numpy as np

def changemode(datalist):
    alphabat = {'S':0, 'M':1, 'L':2}
    for sonlist in datalist:
        sonlist[1] = alphabat(sonlist[1]).value
    return datalist

def loadDataSet():
    """
    加载数据集,数据集有两个特征
    """
    datalist = [[1, 'S'],
                [1, 'M'],
                [1, 'M'],
                [1, 'S'],
                [1, 'S'],
                [2, 'S'],
                [2, 'M'],
                [2, 'M'],
                [2, 'L'],
                [2, 'L'],
                [2, 'L'],
                [3, 'M'],
                [3, 'M'],
                [3, 'L'],
                [3, 'L']]
    
    labels = np.mat([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]).T
    
    #为了便于处理，将第二个特征的字母也映射为数字

    return datalist, labels

if __name__ == "__main__":
    data, labels = loadDataSet()
    changemode(data)