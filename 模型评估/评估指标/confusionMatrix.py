# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 08:05:36 2019
混淆矩阵——利用不平衡数据的结果

@author: Kylin
"""
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. 加载数据,创建一个不平衡数据集
digits = load_digits()
y = digits.target == 9
# 统计下数据中真的为9的样本个数
positive_num = np.sum(y == True)
print("=====原数据集=====")
print(f"数据个数：{len(y)}")
print(f"y为True的个数：{positive_num}")
print(f"真正例比 = {positive_num / len(y)}")

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0, stratify=y)
# X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

# 3. 使用逻辑回归分类器
print("------使用Logistic分类器------")
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("LogisticRegression test score:", logreg.score(X_test, y_test))

# 查看预测结果的数据情况
predict_positive_num = np.sum(pred_logreg == True)
y_test_positive_num = np.sum(y_test == True)
print("=====测试数据集=====")
print(f"数据个数：{len(y_test)}")
print(f"y_test为True的个数：{y_test_positive_num}")
print(f"y_predict为True的个数：{predict_positive_num}")
print(f"真正例比 = {predict_positive_num / len(y_test)}")

# 4. 使用混淆矩阵来查看分类结果
matrix = confusion_matrix(y_test, pred_logreg)
print("混淆矩阵：\n", matrix)
# 参数normalize用于将混淆矩阵归一化
matrix_scaler = confusion_matrix(y_test, pred_logreg, normalize='true')
print("按照真实label进行归一化后的混淆矩阵：\n", matrix_scaler)

