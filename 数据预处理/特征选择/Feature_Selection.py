# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:51:26 2019
特征选择在鸢尾花数据集中的应用
@author: Kylin
"""
from sklearn.datasets import load_iris, load_boston
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression#实际上，导入了两种F检验方法：卡方检验和F回归检验

#1. 特征选择在iris数据集上的应用结果
print("---------iris数据集---------")
iris = load_iris()
X, y = iris.data, iris.target

print("初始数据shape:", X.shape)

#选择与y相关的最重要的k个特征，默认使用f_classif只适用于分类函数，这里使用了卡方检验方法
selection = SelectKBest(chi2, k=2).fit(X,y )
X_new = selection.transform(X)
featureInd = selection.get_support()
print("特征选择后的数据shape:", X_new.shape)
print("选择的特征索引:", featureInd) #可以看到选择了最后两个特征

#2. 特征选择在波士顿房价数据集上的应用结果
print("----------boston房价------------")
boston = load_boston()
X2, y2 = boston.data, boston.target

print("初始数据shape:", X2.shape)

#选择与y相关的最重要的k个特征，默认使用f_regression用于回归数据
selection2 = SelectKBest(f_regression, k=4).fit(X2,y2)
X2_new = selection2.transform(X2)
featureInd2 = selection2.get_support(indices=True)
print("特征选择后的数据shape:", X2_new.shape)
print("选择的特征索引:", featureInd2) #可以看到选择了特征2 5 10 12


