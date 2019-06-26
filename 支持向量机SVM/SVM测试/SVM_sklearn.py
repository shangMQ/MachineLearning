# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 20:33:07 2019
问题：
已知正例点x1=(1,2),x2=(2,3),x3=(3,3),负例点x4=(2,1),x5=(3,2)
试求最大间隔分离超平面和分类决策函数，并在图上画出分离超平面、间隔边界及支持向量
@author: Kylin
"""
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#加载数据
X = np.array([[1,2], [2,3], [3,3], [2,1], [3,2]])
y = np.array([1, 1, 1, -1, -1])
y = y.reshape(-1,1)


#利用数据拟合模型
clf = SVC(kernel='linear', C=10000, gamma=0.1)
clf.fit(X,y)

#获得模型的相关数据
w = (clf.coef_).ravel()#获得w系数
b = clf.intercept_#截距
print("w :", w)
print("b :", b)
support = clf.support_vectors_

#画出图像
#设置中文字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
#绘制原始数据点
plt.scatter(X[:,0], X[:,1], s=50, color='red', label='initial data')

#绘制分隔超平面
k = w[0] / w[1]
t = np.linspace(0, 4, 100)
z = (-b) / w[1] - k * t
plt.plot(t, z, color='k', label="分隔超平面")

#绘制支持向量
plt.scatter(support[:,0], support[:,1], color='r', s=120, label='SV', edgecolors='k')

#绘制分隔边界
b1 = -w[0] * support[0][0] - w[1] * support[0][1]
b2 = -w[0] * support[1][0] - w[1] * support[1][1]
z1 = (-b1) / w[1] - k * t
z2 = (-b2) / w[1] - k * t
plt.plot(t, z1, '--', color='b', label='分隔边界')
plt.plot(t, z2, '--', color='b', label='分隔边界')
plt.title("SVM测试")
plt.legend()
plt.show()
