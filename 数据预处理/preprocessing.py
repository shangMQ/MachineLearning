# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:26:45 2019
利用sklearn提供的预处理函数对数据进行预处理
@author: Kylin
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#1.加载数据
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

#2.构建缩放器
scaler = MinMaxScaler()
scaler.fit(X_train)

#3.变换数据
X_train_scaled = scaler.transform(X_train)


