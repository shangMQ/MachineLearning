# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:54:15 2019
使用SVM对digits数据集进行分类，分别使用三种不同的核宽度gamma参数。
@author: Kylin
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#1. 加载数据,创建一个不平衡数据集
digits = load_digits()
y = digits.target == 9

#2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, 
                                                    random_state=0, stratify=y)

for gamma in [1, 0.05, 0.01]:
    svc = SVC(gamma = gamma).fit(X_train, y_train)
    #计算正确率
    accuracy = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
    print("gamma = {:.2f}, accuracy = {:.2f}, auc = {:.2f}".format(gamma, accuracy, auc))
    plt.plot(fpr, tpr, label="gamma = {:.2f}".format(gamma))

plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("SVM auc with different gamma")
plt.legend()
