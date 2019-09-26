# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:43:09 2019
对数据进行三折划分，利用验证集选定最佳参数后，
利用找到的参数重新构建模型，
同时在训练数据和验证数据上进行训练，
利用尽可能多的数据来构建模型。
@author: Kylin
"""
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 加载数据
iris = load_iris()

#2. 划分数据
#将数据划分为训练集+验证集，测试集
X_trainVal, X_test, y_trainVal, y_test = train_test_split(iris.data, iris.target, random_state=0)
#将数据划分为训练集，验证集
X_train, X_validaton, y_train, y_validation = train_test_split(X_trainVal, y_trainVal, random_state=0)
print("训练集shape:", X_train.shape)
print("验证集shape:", X_validaton.shape)
print("测试集shape:", X_test.shape)

#3. 遍历参数列表，寻找最佳参数
best_validation_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(C = C, gamma = gamma)
        svm.fit(X_train, y_train)
        score = svm.score(X_validaton, y_validation)
        if score > best_validation_score:
            best_validation_score = score
            best_parameters = {"C": C, "gamma": gamma}

print("选择参数:", best_parameters)
print("最佳参数在验证集上的score:", best_validation_score)

#4. 利用最佳参数构建模型
svm = SVC(**best_parameters)
svm.fit(X_trainVal, y_trainVal)
test_score = svm.score(X_test, y_test)
print("测试集精度:", test_score)