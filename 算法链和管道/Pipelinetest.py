# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:43:03 2019
构建管道
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

#4. 之后可以像其他sklearn估计器一样拟合管道
pipe.fit(X_train, y_train)

#5. 查看在测试集上的效果
print("Test score:", pipe.score(X_test, y_test)) 