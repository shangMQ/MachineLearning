# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:02:34 2019
多分类指标
@author: Kylin
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve, accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

#1. 加载数据,创建一个不平衡数据集
digits = load_digits()

#2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, 
                                                    random_state=0)

#3. 构建logistic模型
lr = LogisticRegression().fit(X_train, y_train)

pred = lr.predict(X_test)
print("accuracy score:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

#4. 用热力图绘制混淆矩阵
sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap=plt.cm.gray_r)
plt.xlabel("Predict label")
plt.ylabel("True label")
plt.title("Confusion matrix")

#5. 查看每个类别的precesion\recall\f1score
print(classification_report(y_test, pred))

#6. 多分类的fscore
#micro加权：对每个样本等同看待
print("Micro average f1 score:", f1_score(y_test, pred, average="micro"))

#macro加权：对每个类等同看待，无论类别中的样本数
print("Macro average f1 score:", f1_score(y_test, pred, average="macro"))
