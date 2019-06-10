# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:03:15 2019
在cancer数据集上应用GBDT
@author: Kylin
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

#加载cancer数据集
cancer = load_breast_cancer()

#划分数据集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#构建gbdt模型
gbdt = GradientBoostingClassifier(random_state=0)

#拟合训练集
gbdt.fit(X_train, y_train)

#输出预测结果
train_score = gbdt.score(X_train, y_train)
test_score = gbdt.score(X_test, y_test)
print("-"*5, "未处理的GBDT", "-"*5)
print("训练集精度：{:.3f}".format(train_score))
print("测试集精度：{:.3f}".format(test_score))

#由于训练集精度达到100%，可能存在过拟合，限制大树的深度来加强预剪枝
gbdt2 = GradientBoostingClassifier(random_state = 0, max_depth = 1) #利用深度为1的决策树

#拟合训练集
gbdt2.fit(X_train, y_train)

#输出预测结果
train_score2 = gbdt2.score(X_train, y_train)
test_score2 = gbdt2.score(X_test, y_test)
print("-"*5, "限制最大深度的GBDT", "-"*5)
print("训练集精度：{:.3f}".format(train_score2))
print("测试集精度：{:.3f}".format(test_score2))

#降低学习率来加强预剪枝
gbdt3 = GradientBoostingClassifier(random_state = 0, learning_rate=0.01) #利用学习率为0.01的决策树

#拟合训练集
gbdt3.fit(X_train, y_train)

#输出预测结果
train_score3 = gbdt3.score(X_train, y_train)
test_score3 = gbdt3.score(X_test, y_test)
print("-"*5, "降低学习率的GBDT", "-"*5)
print("训练集精度：{:.3f}".format(train_score3))
print("测试集精度：{:.3f}".format(test_score3))
