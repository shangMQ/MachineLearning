# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:07:30 2019
将基于径向基函数的SVM模型应用到乳腺癌数据上
@author: Kylin
"""
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 加载数据集
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#2. 利用训练集拟合svm模型
svc = SVC()
svc.fit(X_train, y_train)

#3. 输出精度
print("训练集精度:{:.3f}".format(svc.score(X_train, y_train)))
print("测试集精度:{:.3f}".format(svc.score(X_test, y_test)))

#过拟合了，查看每个特征的最大最小值
plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")

#4. 特征在不同的数量集，对SVM模型影响很大
#对数据做预处理
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
#将每个特征值都处理到（0，1）之间
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training

#重新拟合SVM模型
svc = SVC(C=20, gamma=0.1)
svc.fit(X_train_scaled, y_train)

#查看新拟合后模型的精度
print("训练集精度:{:.3f}".format(svc.score(X_train_scaled, y_train)))
print("测试集精度:{:.3f}".format(svc.score(X_test_scaled, y_test)))
