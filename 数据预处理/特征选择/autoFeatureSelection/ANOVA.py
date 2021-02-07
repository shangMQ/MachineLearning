# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:46:30 2019
单变量统计特征选择——ANOVA方差分析
@author: 尚梦琦
"""

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 1. 加载数据集
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print("initial data shape:", X.shape)

# 2. 向原数据集中添加噪声
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(X), 50))
X_withNoise = np.hstack([X, noise])
# 含噪声特征的数据集中前30个特征来自数据集，后50个是我们添加的噪声特征
print("add noise data shape:", X_withNoise.shape)

# 3. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X_withNoise, y, test_size=0.5, random_state=0)

# 4. 使用f_classif和SelectPercentile选择50%的特征
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)

# 对训练集进行变换
X_train_selected = select.transform(X_train)  # 这里会有一个异常提示出现
print("After feature selection data shape:", X_train_selected.shape)

# 5. 查看哪些特征被选中
mask = select.get_support()
print("被选中的特征：", mask)
# 蓝色的表示选中True，白色的表示False即未选中的特征
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
plt.xlabel("Sample index")
plt.show()

# 7. 利用原始数据的全部特征查看拟合效果。
logisticReg = LogisticRegression()
logisticReg.fit(X_train, y_train)
print("=======Initial Data==========")
print("Train Score:", logisticReg.score(X_train, y_train))
print("Test Score:", logisticReg.score(X_test, y_test))

# 8. 利用原始数据选择80%的特征查看拟合效果。
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_sixty = select.transform(X_train)
X_test_sixty = select.transform(X_test)
logisticReg.fit(X_train_sixty, y_train)
print("======60 % Feature Selection Data==========")
print("Train Score:", logisticReg.score(X_train_sixty, y_train))
print("Test Score:", logisticReg.score(X_test_sixty, y_test))

