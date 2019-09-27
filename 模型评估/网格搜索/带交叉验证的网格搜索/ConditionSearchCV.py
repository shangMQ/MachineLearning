# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:43:44 2019
非网格的空间中搜索
SVC有一个kernel参数，其他参数与选择的kernal类型密切相关。
例如，kernel="linear"，那么模型是线性的，只会用到C参数。
如果kernel="rbf"，需要使用C和gamma两个参数。
@author: Kylin
"""
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#1. 加载数据
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

#2. 用字典指定要搜索的参数，考虑到不同核对应的参数不同，将它们放在一个列表字典中
#键是要调节的参数名称，值是尝试的参数设置
param_grid = [{"kernel":["rbf"],"C":[0.001, 0.01, 0.1, 1, 10, 100], "gamma":[0.001, 0.01, 0.1, 1, 10, 100]}, 
              {"kernel":["linear"],"C":[0.001, 0.01, 0.1, 1, 10, 100]}]

#3. 使用SVC模型，要搜索的网格参数param_grid和要使用的交叉验证策略将GridSearchCV实例化
grid_search = GridSearchCV(SVC(), param_grid, cv = 5)

#4. 调用fit方法，对param_grid指定的每种参数组合都运行交叉验证
grid_search.fit(X_train, y_train)

#5. 在测试集上调用score方法
print("test score:", grid_search.score(X_test, y_test))

#6. 最佳参数
print("最佳参数:", grid_search.best_params_)
print("交叉验证的最佳精度:", grid_search.best_score_)

#7. 网格搜索结果
results = pd.DataFrame(grid_search.cv_results_)
print(results.head())

