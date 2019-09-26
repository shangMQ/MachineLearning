# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:59:21 2019
简单网格搜索实现svm参数的寻找
使用两个for循环，对每种参数组合分别训练并评估一个分类器。
@author: Kylin
"""
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 加载数据
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print("训练集shape:", X_train.shape, "测试集shape:", X_test.shape)

#2. 遍历参数列表，寻找最佳参数
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(C = C, gamma = gamma)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {"C": C, "gamma": gamma}

print("选择参数:", best_parameters, "最佳score:", best_score)