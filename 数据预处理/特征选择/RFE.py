# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:57:20 2019
利用Wrapper类型的特征选择方法RFE特征递归消去法选择需要的特征
@author: Kylin
"""
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

#1. 加载数据
data = load_boston()
X, y = data.data, data.target
featurename = data.feature_names
print("源数据shape:", X.shape)
print("feature names:", featurename)

#2.利用LinearRegression作为RFE的模型来选择特征
lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=2)
selectmodel = rfe.fit(X,y)

#3. 输出特征的重要性排序
rank = selectmodel.ranking_
print("特征的重要性排序:")
for i in range(len(featurename)):
    print("({:}, {:})".format(rank[i], featurename[rank[i]-1]))
    
#4. 所选择属性的个数
print("所选择属性的个数:", rfe.n_features_)

