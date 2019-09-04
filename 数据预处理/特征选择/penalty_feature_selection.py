# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:35:08 2019
嵌入型特征选择——利用正则化
根据linearSVC模型来分析特征的重要性
@author: Kylin
"""
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

#1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target
print("原数据shape:", X.shape)

#2. 构造带正则化的线性SVM分类器模型,其中penalty为惩罚方式即正则化方式：l1或者l2，
#dual选择算法以解决双优化或原始优化问题，当样本数大于特征数时，dual=False
#C错误项的惩罚参数
lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X,y)
model = SelectFromModel(lsvc, prefit=True)

#3. 
X_new = model.transform(X)
print("进行特征提取后的数据shape:", X_new.shape)
