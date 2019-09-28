# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 08:05:36 2019
混淆矩阵
利用不平衡数据的结果
@author: Kylin
"""
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings

warnings.filterwarnings("ignore")

#1. 加载数据,创建一个不平衡数据集
digits = load_digits()
y = digits.target == 9

#2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, 
                                                    random_state=0)
#3. 使用逻辑回归分类器
print("------使用Logistic分类器------")
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("LogisticRegression test score:", logreg.score(X_test, y_test))

#4. 使用混淆矩阵来查看分类结果
matrix = confusion_matrix(y_test, pred_logreg)
print("混淆矩阵：\n", matrix)

