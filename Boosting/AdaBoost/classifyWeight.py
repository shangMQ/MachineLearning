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

#设置中文字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.plot(x,y)
plt.title("分类器在不同错误率下的权重图像")
plt.xlabel("错误率")
plt.ylabel("分类器权重")

