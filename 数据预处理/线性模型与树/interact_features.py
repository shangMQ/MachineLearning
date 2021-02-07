# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:38:41 2019
交互特征，在分箱数据信息中加入原始数据特征，构成12维数据集。
@author: Kylin
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
mpl.rcParams[u'axes.unicode_minus'] = False

# 1. 构建数据集
X = np.linspace(-3, 3, 70)
y = 0.5 * X + np.random.randn(1, 70)
test_X = np.linspace(-3, 3, 100).reshape(-1, 1)

# 2. 查看原始数据点分布图像
plt.scatter(X, y, color="k")
plt.xlabel("Input feature")
plt.ylabel("Regression output")
plt.title("Initial Data")
plt.show()

# 3. 特征分箱
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
print("----------特征分箱-----------")
bins = np.linspace(-3, 3, 11)
print("bins:", bins)
which_bin = np.digitize(X, bins=bins)
print("X[:5]:\n", X[:5])
print("相应的箱子编号:\n", which_bin[:5])

# 4. 利用OneHot编码
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print("oneHot_X shape:", X_binned.shape)

# 5. 加入原始特征
X_combined = np.hstack([X, X_binned])
print("X_combined shape:", X_combined.shape)

# 6. 利用线性回归模型拟合数据集
linearReg = LinearRegression().fit(X_combined, y)
print("directly add initial Data trainScore:", linearReg.score(X_combined, y))

testX_binned = encoder.transform(np.digitize(test_X, bins=bins))
testX_combinned = np.hstack([test_X, testX_binned])
plt.plot(test_X, linearReg.predict(testX_combinned), label="LinearRegression Combined")
plt.legend()

# 7. 绘制边界
for bin in bins:
    plt.plot([bin, bin], [-4, 3], ":", c="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# 8. 加入交互特征，构成数据集,有11 + 11 = 22维
X_product = np.hstack([X_binned, X * X_binned])
print("X_product shape:", X_product.shape)

# 9. 利用线性回归模型拟合数据集
linearReg = LinearRegression().fit(X_product, y)
print("add product trainScore:", linearReg.score(X_product, y))

testX_binned = encoder.transform(np.digitize(test_X, bins=bins))
testX_product = np.hstack([testX_binned, test_X * testX_binned])

plt.plot(test_X, linearReg.predict(testX_product), label="LinearRegression Product")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
