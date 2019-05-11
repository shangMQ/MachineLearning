# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:30:09 2019
利用岭回归去掉多余参数，同时提高更好的预测效果。
@author: Kylin
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from OLSregression import loadDataSet

def ridgeRegres(xMat,yMat,λ=0.2):
    """
    计算回归系数
    """
    xTx = xMat.T*xMat
    n = np.shape(xMat)[1]
    denom = xTx + np.eye(n)*λ
    if np.linalg.det(denom) == 0.0:
        #虽然加入岭后，可以解决不可逆的问题。但当λ=0时，仍然可能出现问题
        print("该矩阵为奇异矩阵，不可求逆")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
    
def ridgeTest(xArr,yArr):
    """
    用于在一组λ上测试结果
    """
    xMat = np.mat(xArr)
    yMat=np.mat(yArr).T
    n = np.shape(xMat)[1]#计算特征数
    #首先，进行进行数据标准化操作
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = np.mean(xMat,0)   #calc mean then subtract it off
    xVar = np.var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30 #测试次数
    wMat = np.zeros((numTestPts,n)) 
    for i in range(numTestPts):
        #这里的λ以指数级变化，可以看出λ在取值非常小的值和取值非常大的值对结果造成的影响
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat

if __name__ == "__main__":
    #设置图像的标题字体
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    
    abX, abY = loadDataSet("abalone.txt")
    ridgeweights = ridgeTest(abX, abY)
    print(ridgeweights)
    
    fig = plt.figure("利用岭回归处理特征数过多")
    ax = fig.add_subplot(111)
    for i in range(8):
        λlabel = "特征{:}".format(i+1)
        ax.plot(ridgeweights[:,i], label=λlabel)
        ax.set_xlabel("logλ")
        ax.set_ylabel("特征系数")
    plt.title("特征值系数随λ的变化图像")
    plt.legend()    
    plt.show()
    
