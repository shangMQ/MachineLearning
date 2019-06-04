# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:57:14 2019
将随机森林应用在乳腺癌数据集上
@author: Kylin
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

#加载数据集
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#构建随机森林，利用100棵数集成森林
forest = RandomForestClassifier(n_estimators=100, random_state=2)

#拟合测试集
forest.fit(X_train, y_train)

#在测试集上预测
y_hat = forest.predict(X_test)

#计算准确率
score = np.mean(y_hat == y_test)
print("准确率：", score)
