# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:26:45 2019
利用sklearn提供的预处理函数对数据进行预处理
@author: Kylin
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#1.加载数据
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svm = SVC(C=100, gamma=0.2)

print("-"*10, "使用不同的预处理方式初始化数据对模型预测结果的影响", "-"*10)

#1.利用原始数据直接拟合模型
svm.fit(X_train, y_train) 
result = svm.score(X_test, y_test)
print("使用原始数据的模型预测精度是：{:.2f}".format(result))

#2.构建MinMaxScaler缩放器
scaler = MinMaxScaler()
scaler.fit(X_train)

#变换数据
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#拟合模型
svm.fit(X_train_scaled, y_train)

#评估结果
scaler_result = svm.score(X_test_scaled, y_test)
print("使用MinMaxScaler进行数据预处理之后的模型预测精度是：{:.2f}".format(scaler_result))

#3.构建StandardScaler缩放器
scaler2 = StandardScaler()
scaler2.fit(X_train)

#变换数据
X_train_scaled2 = scaler2.transform(X_train)
X_test_scaled2 = scaler2.transform(X_test)

#拟合模型
svm.fit(X_train_scaled2, y_train)

#评估结果
scaler_result2 = svm.score(X_test_scaled2, y_test)
print("使用StandardScaler进行数据预处理之后的模型预测精度是：{:.2f}".format(scaler_result2))
