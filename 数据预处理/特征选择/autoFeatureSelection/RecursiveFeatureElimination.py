# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:04:30 2019
基于随机森林的迭代特征选择
@author: 尚梦琦
"""
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

#1. 加载数据集
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print("initial data shape:", X.shape)
featureName = cancer.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

#2. 拟合递归特征选择模型RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=27)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)

#3. 查看选中的特征
mask = select.get_support()
plt.matshow(mask.reshape(1,-1), cmap="GnBu")
plt.xlabel("Sample Index")

#4. 性能对比
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Initial test Score:", lr.score(X_test, y_test))

lr.fit(X_train_selected, y_train)
print("Seleted Features test Score:", lr.score(X_test_selected, y_test))

#5. 还可以在RFE内使用的模型来进行预测
print(select.score(X_test, y_test))
