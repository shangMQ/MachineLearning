# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:30:33 2019
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings


#忽略分母为零的警告
warnings.filterwarnings("ignore")   # UndefinedMetricWarning

#错误率
x = np.linspace(0,1,100)
#计算相应的分类器权重
y = 0.5 * np.log2((1-x)/x)

positive_sample =  np.e ** (-1 * y)
negative_sample = np.e ** y

#设置中文字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
fig = plt.figure("分类器权重及样本权重图像")
fig.subplots_adjust(hspace=0.6, wspace=0.4)

ax = fig.add_subplot(1,3,1)
plt.plot(x,y)
plt.title("分类器在不同错误率下的权重图像")
plt.xlabel("错误率")
plt.ylabel("分类器权重")

ax2 = fig.add_subplot(1,3,2)
plt.plot(y, positive_sample)
plt.title("分类正确的样本权重变化")
plt.xlabel("分类器权重")
plt.ylabel("样本权重")

ax3 = fig.add_subplot(1,3,3)
plt.plot(y, negative_sample)
plt.title("分类错误的样本权重变化")
plt.xlabel("分类器权重")
plt.ylabel("样本权重")
