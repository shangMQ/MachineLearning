# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:49:07 2020
马氏距离计算方法仿真
@author: 尚梦琦
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
        
def loadData():
    """加载数据，手动生成一些高斯分布数据"""
    mean = [0, 0]
    cov = [[2, 1], [1, 2]] #协方差
    data = np.random.multivariate_normal(mean, cov, 1000)
    
    return data

def drawPoint(data, title, z=None):
    """绘制原始数据的散点图"""
#    有三个用星号标记点的坐标
    stars = np.array([[3, -2, 0],[3, 2, 0]])
    
    fig = plt.figure(title, figsize=(10,7))
    
    if z is not None:
        data = np.array(data * z)
        stars = np.array(z * stars)
#     1. 绘制原始数据的散点图
    plt.scatter(data[:,0], data[:,1], s=30)
#     2. 绘制星星
    plt.scatter(stars[0,:], stars[1,:],s=200, marker="*", color="r")
      
#     3. 绘制坐标轴
    plt.axhline(linewidth=2, color="g") #x轴
    plt.axvline(linewidth=2, color="g") #y轴
          
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    
 
def drawCholeskyTransPlot(data):
    """
    绘制经过Cholesky转换的数据图像
    Cholesky分解就是把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积
    """
#    求X和Y的协方差矩阵
    covArray = np.matrix(np.cov(data[:,0],data[:,1]))
#    求仿射矩阵
    Z = np.linalg.cholesky(covArray).I
    
    drawPoint(data, "CholeskyTransform", Z)
    
    return covArray, Z
    

if __name__ == "__main__":
#   1.加载数据
    data = loadData()
    
#   2. 查看数据散点图
    drawPoint(data, "Initial Data")

#   3. 查看数据经过Cholesky变换后的散点图
    covArray, Z = drawCholeskyTransPlot(data)

#   4. 计算马氏距离
    m_dist1 = dist.mahalanobis([0, 0], [3, 3], covArray.I)
    m_dist2 = dist.mahalanobis([0,0], [-2,2], covArray.I)
    print("(3,3)到原点的马氏距离是:", m_dist1)
    print("(-2,2)到原点的马氏距离是:", m_dist2)
    
#    5. 计算变换后的欧几里得距离
    stars = np.array([[3, -2, 0],[3, 2, 0]])
    dots = Z * stars
    E_dist1 = dist.minkowski([0,0], dots[:,0])
    E_dist2 = dist.minkowski([0,0], dots[:,1])
    print("(3,3)经过Cholesky变换后到原点的欧式距离是:", E_dist1)
    print("(-2,2)经过Cholesky变换后到原点的欧式距离是:", E_dist2)
    
    
    

