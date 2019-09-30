# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:41:26 2019
常规方法：
划分数据集，
使用MinMaxScaler()对数据进行缩放,
然后拟合SVM模型用于预测cancer
@author: Kylin
"""
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

#3. 对数据进行MinMax缩放
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#找到更好的参数
param_grid = {"C": [0.001, 0.01, 0.1, 10, 100], "gamma":[0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

print("Best cross_validation accuracy:", grid.best_score_)
print("Best set score:", grid.score(X_test_scaled, y_test))
print("Best parameters:", grid.best_params_)
"""
#4. 构建SVM模型
svm = SVC()
svm.fit(X_train_scaled, y_train)


#5. 查看测试集上的score
print("Test score:", svm.score(X_test_scaled, y_test))
"""