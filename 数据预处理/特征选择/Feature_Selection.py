# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:51:26 2019
特征选择在鸢尾花数据集中的应用
@author: Kylin
"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

print("初始数据shape:", X.shape)

#选择与y相关的最重要的k个特征，默认使用f_classif只适用于分类函数
selection = SelectKBest(chi2, k=2).fit(X,y )
X_new = selection.transform(X)
featureInd = selection.get_support()
print("特征选择后的数据shape:", X_new.shape)
print("选择的特征索引:", featureInd) #可以看到选择了最后两个特征

