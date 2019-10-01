# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:47:36 2019
访问网格搜索管道中的属性
@author: Kylin
"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

#1. 加载数据
cancer = load_breast_cancer()

#2. 构建管道
pipe = make_pipeline(StandardScaler(), LogisticRegression())

#3. 构建参数网格(注意名称)
param_grid = {"logisticregression__C":[0.01, 0.1, 1, 10, 100]}

#4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4)

#5. 网格搜索
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

#6. gridSearch找到的最佳模型保存在best_estimator_中
print("Best Estimator:", grid.best_estimator_)

#7. 使用pipe.named_steps属性来访问Logisticregression步骤
print("LogisticRegression step:\n", grid.best_estimator_.named_steps["logisticregression"])

#8. 访问与每个输入特征相关的系数权重
print("Logistic regression coefficients:", grid.best_estimator_.named_steps["logisticregression"].coef_)