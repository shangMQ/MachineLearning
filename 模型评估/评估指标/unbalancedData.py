# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:27:39 2019
不平衡数据集的处理
@author: Kylin
"""
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. 加载数据,创建一个不平衡数据集
digits = load_digits()
y = digits.target == 9

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, 
                                                    random_state=0, stratify=y)

# 3. 使用DummyClassifier来始终预测多数类
print("-----使用Dummy分类器（最常见预测）-------")
dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
predict = dummy_majority.predict(X_test)
print("Unique predicted labels:", np.unique(predict))
print("Test score:", dummy_majority.score(X_test, y_test))

# 4. 使用决策树分类器
print("------使用决策树分类器------")
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("Unique predicted labels:", np.unique(pred_tree))
print("Test score:", tree.score(X_test, y_test))

# 5. 使用默认的Dummy分类器
print("------使用默认的Dummy分类器（随机预测）------")
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("Dummy test score:", dummy.score(X_test, y_test))

# 6. 使用逻辑回归分类器
print("------使用Logistic分类器------")
logreg = LogisticRegression(C=0.1, max_iter=1000).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("LogisticRegression test score:", logreg.score(X_test, y_test))

# 7. 使用混淆矩阵来查看分类结果
print("---------使用混淆矩阵评价分类结果----------")
matrix1 = confusion_matrix(y_test, predict)
print("使用Dummy分类器（最常见预测)混淆矩阵：\n", matrix1)

matrix2 = confusion_matrix(y_test, pred_tree)
print("决策树分类器 混淆矩阵：\n", matrix2)

matrix3 = confusion_matrix(y_test, pred_dummy)
print("使用Dummy分类器（随机预测)混淆矩阵：\n", matrix3)

matrix4 = confusion_matrix(y_test, pred_logreg)
print("使用LogisticRegression分类器 混淆矩阵：\n", matrix4)

# 8. 使用f1分数评价分类结果
print("---------使用f1分数评价分类结果-----------")
print("使用Dummy最常见预测的f1-score:", f1_score(y_test, predict))

print("使用决策树的f1-score:", f1_score(y_test, pred_tree))

print("使用Dummy随机预测的f1-score:", f1_score(y_test, pred_dummy))

print("使用LogisticRegression的f1-score:", f1_score(y_test, pred_logreg))

# 9. 使用classification_report直接计算precision、recall、f1score
print("---------使用classification_report评价分类结果-----------")
print(classification_report(y_test, pred_logreg, target_names=["not nine", "nine"]))

