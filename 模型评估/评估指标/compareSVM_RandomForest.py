# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:26:55 2019
比较SVM和RandomForestClassifier分类器
@author: Kylin
"""
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#1. 加载数据集
X,y = make_blobs(n_samples=4500, centers=2, cluster_std=[7, 2], random_state=22)

#2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#3. 拟合SVC模型
svc = SVC(gamma=0.5).fit(X_train, y_train)

#4. 绘制precision_recall_curve
#需要两个参数：真实标签，预测的不确定度，可以有decision_function和predict_proba给出。
#之后会返回一个列表，包含按顺序排序的所有可能阈值对应的准确率和召回率，然后就可以绘制一条曲线。
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

#找到最接近0的阈值下标
close_zero = np.argmin(np.abs(thresholds))

#将最小阈值对应的precesion\recall值绘制
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c="k", mew=2)

plt.plot(precision, recall, label="SVM precision recall curve")

#5. 拟合RandomForest模型
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

#randomForestClassifier没有decision_function，只有predict_proba.
#第二个参数应该是正类别的确定性度量，所以传入样本属于类别1的概率，二分类默认阈值时0.5
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:,1])

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))

plt.plot(precision[close_default_rf], recall[close_default_rf], '^', markersize=10,
         label="threshold rf", fillstyle="none", c="k", mew=2)
plt.plot(precision_rf, recall_rf, label="RandomForestClassifier precision recall curve")
plt.xlabel("precesion")
plt.ylabel("recall")
plt.title("precision recall curve")
plt.legend()

#6. f1score对比
#f1score只反映了准确率-召回率曲线上的一个点，也就是默认阈值对应的那个点。
print("SVM F1-score:", f1_score(y_test, svc.predict(X_test)))
print("RandomForest F1-score:", f1_score(y_test, rf.predict(X_test)))

#7. 计算曲线下的面积
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:,1])
print("average precision of svc:", ap_svc)
print("average precision of randomforest:", ap_rf)

