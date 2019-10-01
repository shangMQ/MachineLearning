# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:16:33 2019
网格搜索预处理步骤与模型参数
@author: Kylin
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

#1. 加载数据
boston = load_boston()

#2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

#3. 构建管道
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

#4. 如何选择多项式次数呢？想要根据分类结果来选择degree参数
param_grid = {"polynomialfeatures__degree":[1,2,3],
              "ridge__alpha":[0.001, 0.01, 0.1, 1, 10, 100]}

#5. 进行网格搜索
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

#6. 利用热力图将交叉验证的结果可视化
plt.matshow(grid.cv_results_["mean_test_score"].reshape(3,-1), vmin=0, cmap="viridis")
plt.xlabel("ridge__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.xticks(range(len(param_grid["ridge__alpha"])), param_grid["ridge__alpha"])
plt.yticks(range(len(param_grid["polynomialfeatures__degree"])), param_grid["polynomialfeatures__degree"])
plt.title("GridSearchCV result")
plt.colorbar()
plt.show()

#7. 查看最佳参数
print("best parameters:", grid.best_params_)
print("Test score:", grid.score(X_test, y_test))

#9. 运行一个没有多项式特征的网格搜索
param_grid = {"ridge__alpha":[0.001, 0.01, 0.1, 1, 10, 100]}
pipe2 = make_pipeline(StandardScaler(), Ridge())
grid2 = GridSearchCV(pipe2, param_grid, cv=5)
grid2.fit(X_train, y_train)
print("Test score without poly features:", grid2.score(X_test, y_test))