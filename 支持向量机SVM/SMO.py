# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:31:21 2019
完整版SMO
与简版不同之处，在于选择αj的方式。
@author: 尚梦琦
"""

import numpy as np

def selectJrand(i, m):
    """
    在某个区间范围内随机选择一个整数
    参数：α的下标i，α的总个数m
    返回值：为了与i构成α对而随机选择一个下标j
    """
    mlist = [j for j in range(m)]
    del mlist[i]
    index = int(random.uniform(0, m-1))
    return mlist[index]

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.toler = toler
        self.m, self.n = np.shape(dataMatIn)
        self.alphas = np.mat(np.zeros((self.m, 1))) #存储α的值
        self.Ecache = np.mat(np.zeros((self.m, 2))) #存储误差E
        #其中Echache的第一列给出的是Echache是否有效的标志位，
        #第二列给出的是实际的E值
        
    def clacEk(oS, k):
        """
        计算第k个样本的误差
        """
        fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
        Ek = fXk - oS.labelMat[k]
        return Ek
    
    def selectJ(i, oS, Ei):
        """
        内循环中的启发式方法，选择与αi配对的αj
        目标：选择第二个α值以保证每次优化中采用最大步长
        参数：第一个α值的下标i，误差Ei
        返回值：选择的αj的下标j，第j个样本的误差Ej
        """
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        #首先将Ei设置为有效的（表示已经计算好了）
        oS.Ecache[i] = [1, Ei]
        #np.nonzero()返回一个列表，这个列表中包含以输入列表为目录的列表值，返回的是非零E值所对应的α值
        validEchacheList = np.nonzero(oS.Ecache[:,0].A)[0]
        
        if len(validEchacheList) > 1:
            #如果有效Cache列表的长度大于1，则选择一个具有最大步长的j
            for k in validEchacheList:
                if k == i:
                    continue
                Ek = oS.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
                
        else:#否则从所有样本中随机选择一个与αi成对
            j = selectJrand(i, oS.m)
            Ej = oS.clacEk(oS, j)
        
        return j, Ej
    
    def updateEk(oS, k):
        """
        计算误差值并存入Ecache中
        """
        Ek = oS.clacEk(oS, k)
        oS.Ecache[k] = [1, Ek]
            
                
        
        