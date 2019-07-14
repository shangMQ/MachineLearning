# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:58:05 2019
对训练数据和测试数据使用相同的缩放
@author: Kylin
"""
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#1.加载数据
X,_ = make_blobs(n_samples = 50, centers=5, random_state=4, cluster_std=2)

#2.划分数据集
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

#3.绘制训练集和测试集
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.subplots_adjust(hspace=0.6, wspace=0.3)
#绘制原始数据散点图
axes[0].scatter(X_train[:, 0], X_train[:, 1], label="Training set", s = 60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', label="Test set", s = 60)
axes[0].set_title("Initial Data")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].legend(loc="upper left")

#利用MinMaxScaler缩放数据
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], label="Training set", s = 60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', label="Test set", s = 60)
axes[1].set_title("MinMaxScaler Data")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend(loc="upper left")


#利用StandardScaler缩放数据
scaler2 = StandardScaler()
scaler2.fit(X_train)
X_train_scaled2 = scaler2.transform(X_train)
X_test_scaled2 = scaler2.transform(X_test)
axes[2].scatter(X_train_scaled2[:, 0], X_train_scaled2[:, 1], label="Training set", s = 60)
axes[2].scatter(X_test_scaled2[:, 0], X_test_scaled2[:, 1], marker='^', label="Test set", s = 60)
axes[2].set_title("StandardScaler Data")
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
axes[2].legend(loc="upper left")

fig.suptitle("Using different preprocessing ways initialize data")