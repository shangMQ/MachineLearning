# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:51:16 2019
打乱划分交叉验证
每次划分为训练集取样train_size个点，为测试集取样test_size个（不相交）的点。
将这一划分方法重复n_iter次。
@author: Kylin
"""
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

#1. 加载数据
iris = load_iris()

#2. logistic回归模型
logreg = LogisticRegression()

#3. ShuffleSplit
shuffle = ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=10)

#4、交叉验证模型
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle)
print("得分：", scores)
print("均值：", scores.mean())


