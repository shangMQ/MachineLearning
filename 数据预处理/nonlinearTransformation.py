# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:33:49 2019
单变量非线性变换的模拟
@author: Kylin
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
mpl.rcParams[u'axes.unicode_minus'] = False

# 1. 产生数据集
random = np.random.RandomState(0)
X_org = random.normal(size=(1000, 3))
w = random.normal(size=3)

X = random.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

# 2. 查看第一个特征的数值分布
# print("特征1的数值分布情况:\n", np.bincount(X[:,0]))
bins = np.bincount(X[:,0])
plt.bar(range(len(bins)), bins, color="k")
plt.ylabel("Number of appearance")
plt.xlabel("Value")
plt.xlim(0, 140)
plt.title("The appearance number of values")
plt.show()

# 3. 利用岭回归直接拟合原始的泊松分布数据
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
initialScore = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("Initial Data Score:", initialScore)

# 4. 对数据进行对数变换后重新使用岭回归拟合
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

# 对变换后的数据进行可视化
fig = plt.figure()
plt.hist(X_train_log[:,0], bins=25, color="gray", edgecolor='k')
plt.ylabel("Number of appearances")
plt.xlabel("Value")
plt.xlim(0, 5)
plt.title("Log transformation data")
plt.show()

# 重新拟合岭回归模型
TransformScore = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("Log Transformation Data Score:", TransformScore)
