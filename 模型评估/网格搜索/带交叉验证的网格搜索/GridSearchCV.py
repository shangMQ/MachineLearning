# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:48:52 2019
带交叉验证的网格搜索
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

#2. 用字典指定要搜索的参数
#键是要调节的参数名称，值是尝试的参数设置
param_grid = {"C":[0.001, 0.01, 0.1, 1, 10, 100], "gamma":[0.001, 0.01, 0.1, 1, 10, 100]}

#3. 使用SVC模型，要搜索的网格参数param_grid和要使用的交叉验证策略将GridSearchCV实例化
grid_search = GridSearchCV(SVC(), param_grid, cv = 5)

#4. 调用fit方法，对param_grid指定的每种参数组合都运行交叉验证
grid_search.fit(X_train, y_train)

#5. 在测试集上调用score方法
print("test score:", grid_search.score(X_test, y_test))

#6. 最佳参数
print("最佳参数:", grid_search.best_params_)
print("交叉验证的最佳精度:", grid_search.best_score_)

#7. 查看最佳参数对应的模型
print("最佳参数对应的模型:", grid_search.best_estimator_)

#8. 网格搜索结果
results = pd.DataFrame(grid_search.cv_results_)
print(results.head())

#9. 提取平均验证分数，然后改变分数数组的形状，使其坐标轴分别对应于C和gamma
scores = np.array(results.mean_test_score).reshape(6,6)

#10. 利用热力图将二维参数网格可视化
xticks = param_grid["gamma"]
yticks = param_grid["C"]

sns.heatmap(scores, xticklabels=xticks, yticklabels=yticks, cmap="viridis", annot=True)
plt.xlabel("gamma")
plt.ylabel("C")
plt.title("Two dimension parameters grid")