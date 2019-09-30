# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:59:54 2019
在网格搜索中使用管道
现在对于交叉验证的每次划分来说，仅使用训练部分对MinMaxScaler进行拟合，
测试部分的信息没有泄露到参数搜索中。
@author: Kylin
"""
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

#1. 加载数据集
cancer = load_breast_cancer()

#2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#3. 构建管道(每个步骤都是一个元组，其中包含一个名称和一个估计器对象)
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])

#4. 网格参数(每个参数前面都要加上相应的步骤名称和双下划线__)
param_grid = {"svm__C":[0.001, 0.01, 0.1, 1, 10, 100],
              "svm__gamma":[0.001, 0.01, 0.1, 1, 10, 100]}

#5. 使用GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross_validation accuracy:", grid.best_score_)
print("Best set score:", grid.score(X_test, y_test))
print("Best parameters:", grid.best_params_)
