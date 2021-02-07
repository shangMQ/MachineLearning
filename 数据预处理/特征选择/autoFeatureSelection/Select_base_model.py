# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:27:32 2019
基于模型的特征选择
@author: 尚梦琦
"""
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# 1. 加载数据集
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print("initial data shape:", X.shape)
featureName = cancer.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# 2. 构建基于随机森林的特征模型选择器
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), 
                         threshold="median")
select.fit(X_train, y_train)
X_train_selectFeature = select.transform(X_train)
print("X_train shape:", X_train.shape)
print("特征选择后的shape:", X_train_selectFeature.shape)

# 3. 查看选中的特征
mask = select.get_support()
selectFeatures = featureName[mask == True]
print("选中的属性名：\n", selectFeatures)
plt.matshow(mask.reshape(1, -1), cmap="GnBu")
plt.xlabel("Sample Index")
plt.show()

# 4. 性能对比
lr1 = LogisticRegression()
lr1.fit(X_train, y_train)

print("Initial train Score:", lr1.score(X_train, y_train))
print("Initial test Score:", lr1.score(X_test, y_test))

lr2 = LogisticRegression()
lr2.fit(X_train_selectFeature, y_train)
X_test_selectFeature = select.transform(X_test)
print("Selected Features train Score:", lr2.score(X_train_selectFeature, y_train))
print("Selected Features test Score:", lr2.score(X_test_selectFeature, y_test))
