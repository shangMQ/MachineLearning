# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:24:24 2019
分组交叉验证
适用于数据中的分组高度相关时，可以用GroundKFold，以groups数组作为参数。
groups表示数据中的分组，在创建训练集和测试集的时候不应该将其分开。
对于每次划分，每个分组都是整体出现在训练集或者测试集中。
@author: Kylin
"""
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
import warnings

warnings.filterwarnings("ignore")

#1. 生成数据
X,y = make_blobs(n_samples=12, random_state=0)

#2. 分组【3，4，5】
groups = [0, 0, 0, 1, 1, 1, 1, 5, 5, 5, 5, 5]

#3. Logistic回归模型
logreg = LogisticRegression()

#4. 利用分组交叉验证
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print(scores)

