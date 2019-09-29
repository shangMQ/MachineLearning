# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 23:06:36 2019
用AUC分数对digits数据集中“9与其他”任务上的SVM进行评估，将默认的精度修改为AUC指标。
@author: Kylin
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics.scorer import SCORERS
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

#1. 加载数据,创建一个不平衡数据集
digits = load_digits()

#2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target == 9, 
                                                    random_state=0)

#3. 改变cross_val_score的评估指标
#默认使用精度评估
print("默认精度：", cross_val_score(SVC(), digits.data, digits.target == 9))

#使用AUC评估
print("AUC评估：", cross_val_score(SVC(), digits.data, digits.target == 9, scoring="roc_auc"))

#4. GridSearchCV用于选择最佳参数的默认指标
param_grid = {"gamma":[0.0001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
print("----使用默认的策略accuracy进行网格搜索----")
print("最佳参数:", grid.best_params_)
print("最佳交叉验证score:", grid.best_score_)
print("测试集上的AUC:", roc_auc_score(y_test, grid.decision_function(X_test)))
print("测试集score:", grid.score(X_test, y_test))

#5. 改变GridSearchCV用于选择最佳参数的指标，改为AUC
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
grid.fit(X_train, y_train)
print("----使用AUC策略进行网格搜索----")
print("最佳参数:", grid.best_params_)
print("最佳交叉验证score:", grid.best_score_)
print("测试集上的AUC:", roc_auc_score(y_test, grid.decision_function(X_test)))
print("测试集score:", grid.score(X_test, y_test))

