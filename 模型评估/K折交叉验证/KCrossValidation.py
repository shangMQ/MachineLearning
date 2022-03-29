# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:23:23 2019
在鸢尾花数据集上的对LogistRegression模型进行K折交叉验证
@author: Kylin
"""
from sklearn.model_selection import cross_val_score, KFold
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
print("--------5 fold cross validation------")
print("Cross Validation scores:", scores)
print("Five Cross Validation Mean score:", scores.mean())

#4. 使用交叉验证分离器KFold作为cv的参数
kfold = KFold(n_splits=3)
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("--------use 3 KFold(其实就是鸢尾花数据上不分层) cross validation------")
print("Cross Validation scores:", scores)
print("Five Cross Validation Mean score:", scores.mean())
print("原因是因为原始的鸢尾花数据每一折对应一个类别")

#4. 使用交叉验证分离器KFold作为cv的参数
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("--------use 3 KFold(数据划分之前将其打乱) cross validation------")
print("Cross Validation scores:", scores)
print("Five Cross Validation Mean score:", scores.mean())
