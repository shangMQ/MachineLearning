# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:52:00 2019
决策树实现乳腺癌分类
@author: Kylin
"""
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
import numpy as np


def plot_feature_importances_cancer(model):
    fig = plt.figure("特征重要性")
    n_features = cancer.data.shape[1]
    #绘制水平条形图
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

#1. 导入数据集
cancer = load_breast_cancer()
#将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    stratify=cancer.target, random_state=42)

#2. 构建决策树模型
tree = DecisionTreeClassifier(random_state = 0, max_depth=4)

#3. 利用训练集拟合模型
tree.fit(X_train, y_train)

#4. 查看拟合效果
train_score = tree.score(X_train, y_train) #训练集分类正确率
test_score = tree.score(X_test, y_test) #测试集分类正确率

print("training set accuracy : {:.2f}".format(train_score))
print("test set accuracy : {:.2f}".format(test_score))

#5. 分析决策树
export_graphviz(tree, out_file="cancerTree.dot", class_names=["malignant","benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
with open("cancerTree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

#6. 树的特征重要性
print("Feature importance:")
print(tree.feature_importances_)
#可视化
plot_feature_importances_cancer(tree)

