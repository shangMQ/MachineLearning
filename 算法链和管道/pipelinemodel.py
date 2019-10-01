# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:01:44 2019
Pipeline原理简析
@author: Kylin
"""

def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        #遍历除最后一步之外的所有步骤
        X_transformed = estimator.fit_transform(X_transformed, y)
    
    #拟合最后一个estimator
    self.steps[-1][1].fit(X_transformed)
    return X_transformed

def predict(self, X):
    X_transformed = X
    for step in self.steps[-1]:
        #对除了最后一步之外的所有步骤进行变换
        X_transformed = step[1].transform(X_transformed)
    
    #利用最后一个estimator做预测
    return self.steps[-1][1].predict(X_transformed)
