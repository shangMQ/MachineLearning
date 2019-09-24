# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:39:42 2019
留一法交叉验证
每折只包含一个样本的交叉验证，对于每次划分，选择单个样本点作为测试集。
优缺点：
    耗时，在小型数据集上有时会给出更好的估计结果。
@author: Kylin
"""
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

#1. 加载数据
iris = load_iris()

#2. logistic回归模型
logreg = LogisticRegression()

#3、交叉验证模型
leave = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=leave)
print("得分个数：", len(scores))
print("均值：", scores.mean())


