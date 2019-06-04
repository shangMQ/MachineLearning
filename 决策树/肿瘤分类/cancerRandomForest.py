# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:57:14 2019
将随机森林应用在乳腺癌数据集上
@author: Kylin
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances_cancer(model):
    #可视化特征重要性
    fig = plt.figure("特征重要性")
    n_features = cancer.data.shape[1]
    #绘制水平条形图
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("the features importance by using randomforest")

#加载数据集
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#构建随机森林，利用100棵数集成森林
forest = RandomForestClassifier(n_estimators=100, random_state=2)

#拟合测试集
forest.fit(X_train, y_train)

#在测试集上预测
y_hat = forest.predict(X_test)

#计算准确率
score = np.mean(y_hat == y_test)
print("准确率：", score)


#可视化特征重要性,一般来说随机森林给出的特征重要性比单棵树给出的更为可靠
plot_feature_importances_cancer(forest)
