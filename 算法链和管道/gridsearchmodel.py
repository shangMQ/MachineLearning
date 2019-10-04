#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:58:00 2019
在iris数据集上网格搜索比较使用哪个模型
@author: Kylin
"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
import warnings

warnings.filterwarnings("ignore")

#1. 加载数据集
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

#2. 构建管道
pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])
param_grid = [{'classifier':[SVC()],'preprocessing': [StandardScaler(), None],'classifier__gamma':[0.001, 0.01, 0.1, 1, 10, 100],'classifier__C':[0.001, 0.01, 0.1, 1, 10, 100]},
              {'classifier':[RandomForestClassifier(n_estimators=100)],'preprocessing':[None], 'classifier__max_features':[1, 2, 3]}]

#3. 网格搜索
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best params:\n", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)
print("test score:", grid.score(X_test, y_test))
