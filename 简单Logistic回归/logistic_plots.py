# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:44:06 2019
μ参数控制对称中心，函数以（μ，0.5）对称
γ控制函数图像的形状，γ值越小，曲线在中心附近的增长越快
@author: Kylin
"""
import matplotlib.pyplot as plt
import numpy as np

def function(x, μ, γ):
    #产生逻辑斯蒂回归函数的值
    z = np.e ** (-(x - μ)/γ)
    y = 1.0 / (1 + z)
    return y    

x = np.linspace(-10,10,1000)

#设置不同的参数取值
γ = [0.1, 0.5, 1]
μ = [0, 1, 2] 

k = 1

fig = plt.figure("logistic的参数的取值对图像的影响")
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,k)
        ax = fig.add_subplot(3,3,k)
        y = function(x, μ[i], γ[j])
        linelabel = "μ = {:} γ = {:}".format(μ[i], γ[j])
        plt.plot(x,y)
        plt.scatter(μ[i], 0.5, color='r')
        plt.title(linelabel)
        k += 1
        