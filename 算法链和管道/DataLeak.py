# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:33:04 2019
在特征提取和特征选择中，如果使用测试部分，会带来什么样的后果
@author: Kylin
"""
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

#1. 生成数据
rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100,))
#考虑到X和y是独立的，所以应该不可能从数据集中学到任何内容

#2. 使用SelectPercentile特征选择从10000个特征中选择信息量最大的特征
select = SelectPercentile(score_func=f_regression, percentile=5).fit(X,y)
X_selected = select.transform(X)
print("X_selected shape:", X_selected.shape)

#3. 使用交叉验证对Ridge回归进行评估
score = np.mean(cross_val_score(Ridge(), X_selected, y, cv=5))
print("Cross Validation score:", score)

#4. 正确做法（使用Pipeline）
pipe = Pipeline([("select", SelectPercentile(score_func=f_regression, percentile=5)),
                 ("ridge", Ridge())])
score = np.mean(cross_val_score(pipe, X, y, cv=5))
print("pipeline cross validation score:", score)