# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:59:54 2019
在网格搜索中使用管道
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