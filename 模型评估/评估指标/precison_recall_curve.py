# -*- coding: utf-8 -*-
"""
查准率（横轴）-召回率（纵轴）曲线  PR图
根据学习器的预测结果，对样例进行排序，排在前面的是学习器认为最有可能是正例的样本，排在最后的则是学习器认为最不可能是正例的样本。
按照顺序逐个把样本作为正例进行预测，每次都可以计算当前的Precision和recall值。

同时查看所有可能的阈值或准确率和召回率的所有可能折中
@author: Kylin
"""
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

# 1. 加载数据集
X, y = make_blobs(n_samples=4500, centers=2, cluster_std=[7, 2], random_state=22)

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 可视化数据集
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=0.7)
plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.title("Data(train)")
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.title("Data(test)")
plt.suptitle("Data show")
plt.show()

# 3. 拟合SVC模型
svc = SVC(gamma=0.5).fit(X_train, y_train)

# 4. 绘制precision_recall_curve
# 需要两个参数：真实标签，预测的不确定度，可以有decision_function和predict_proba给出。
# 之后会返回一个列表，包含按顺序排序的所有可能阈值对应的准确率和召回率，然后就可以绘制一条曲线。
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

# 找到最接近0的阈值下标
close_zero = np.argmin(np.abs(thresholds))

# 将最小阈值对应的precesion\recall值绘制
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c="k", mew=2)
plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("precesion")
plt.ylabel("recall")
plt.title("precision recall curve")
plt.legend()
plt.show()
