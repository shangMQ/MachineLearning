# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:30:24 2019
不确定性问题
@author: Kylin
"""
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC

#1. 加载数据集
X,y = make_blobs(n_samples=450, centers=2, cluster_std=[7, 2], random_state=22)

#可视化数据集
colors = ["r", "b"]
X1 = X[y == 0]
X2 = X[y == 1]
plt.scatter(X1[:,0], X1[:,1], color=colors[0], label="Negative Class")
plt.scatter(X2[:,0], X2[:,1], color=colors[1], label="Positive Class")
plt.title("Initial Data Fig")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.legend()


#2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#3. 使用SVM模型分类
svc = SVC(gamma=0.5).fit(X_train, y_train)
pred = svc.predict(X_test)

#4. 使用classification_report评估分类结果
print(classification_report(y_test, pred))