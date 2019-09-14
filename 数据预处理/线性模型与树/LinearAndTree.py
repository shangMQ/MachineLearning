# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:03:16 2019
比较线性回归和决策树
@author: Kylin
"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

#1. 构建数据集
X = np.linspace(-3,3,70)
y = 0.5 * X + np.random.randn(1,70)
test_X = np.linspace(-3,3,100).reshape(-1,1)

#2. 查看原始数据点分布图像
plt.scatter(X, y, color="k")
plt.xlabel("Input feature")
plt.ylabel("Regression output")

#3. 利用线性回归模型
print("-------LinearRegression---------")
X = X.reshape(-1,1)
y = y.reshape(-1,1)
linearReg = LinearRegression()
linearReg.fit(X,y)
print("斜率:", linearReg.coef_.ravel()[0], "截距:", linearReg.intercept_[0])

y_predict = linearReg.predict(test_X)
plt.plot(test_X.ravel(), y_predict.ravel(), color="r", label="linearRegression")
print("trainScore：{:.2f}".format(linearReg.score(X, y)))


#4. 利用决策树
print("-----DescionTreeRregression-----")
decisionTree = DecisionTreeRegressor(min_samples_split=3).fit(X,y)
y_predict2 = decisionTree.predict(test_X)

plt.plot(test_X.ravel(), y_predict2.ravel(), color="b", label="DecisionTreeRegression")
print("trainScore：{:.2f}".format(decisionTree.score(X, y)))


#4. 特征分箱
print("----------特征分箱-----------")
bins = np.linspace(-3,3,11)
print("bins:", bins)
which_bin = np.digitize(X, bins=bins)
print("X[:5]:\n", X[:5])
print("相应的箱子编号:\n", which_bin[:5])

#5. 利用OneHot编码
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print("oneHot_X shape:", X_binned.shape)

#6. 重新利用模型拟合数据
print("-------LinearRegression(One hot data)-----")
testX_binned = encoder.transform(np.digitize(test_X, bins=bins))
linearReg = LinearRegression()
linearReg.fit(X_binned,y)
print("斜率:", linearReg.coef_.ravel()[0], "截距:", linearReg.intercept_[0])

y_predict = linearReg.predict(testX_binned)
plt.plot(test_X.ravel(), y_predict.ravel(), color="y", label="linearRegression Binned")
print("trainScore：{:.2f}".format(linearReg.score(X_binned, y)))

print("-----DescionTreeRregression(One hot data)-----")
decisionTree = DecisionTreeRegressor(min_samples_split=3).fit(X_binned,y)
y_predict2 = decisionTree.predict(testX_binned)

plt.plot(test_X.ravel(), y_predict2.ravel(), color="pink", linestyle='--',label="DecisionTreeRegression Binned")
print("trainScore：{:.2f}".format(decisionTree.score(X_binned, y)))
plt.vlines(bins, -3, 3, linewidth=1, alpha=0.2)
plt.legend()