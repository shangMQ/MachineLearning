# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:23:23 2019
在鸢尾花数据集上的对LogistRegression模型进行K折交叉验证
@author: Kylin
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import warnings

#忽略异常
warnings.filterwarnings("ignore")

#1. 获取鸢尾花数据
iris = load_iris()

#2. 逻辑回归模型
logreg = LogisticRegression()

#3. 利用五折交叉验证查看逻辑回归模型的精度
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Cross Validation scores:", scores)

#4、计算平均值
print("Five Cross Validation Mean score:", scores.mean())
