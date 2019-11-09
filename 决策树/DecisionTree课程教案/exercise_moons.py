# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:24:42 2019
利用moons数据集训练和微调决策树模型
@author: Kylin
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def plotdata(X,y):
    class1 = X[(y == 0)]
    class2 = X[(y == 1)]
    plt.scatter(class1[:,0], class1[:,1], color="red", label="class1")
    plt.scatter(class2[:,0], class2[:,1], color="green", label="class2")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.title("Moon dataset")

#1. 生成moons数据
X,y = make_moons(n_samples=10000, noise=0.4)

#2. 可视化数据集
plotdata(X,y)

#3. 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. 构建参数列表字典
params = {'max_leaf_nodes': list(range(2, 100)),
          'min_samples_split': [2, 3, 4]}

#5. 使用GridSearch和3折交叉验证找到最佳超参数
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
print("最佳超参数模型：", grid_search_cv.best_estimator_)
print("最佳超参数：",grid_search_cv.best_params_)

#默认情况下，GridSearchCV会训练在整个训练集上找到的最佳模型（您可以通过设置refit = False来更改此模型），因此我们无需再次进行此操作。
#6. 利用测试集评估模型好坏
predict_score = grid_search_cv.score(X_test, y_test)
print("test score:", predict_score)

