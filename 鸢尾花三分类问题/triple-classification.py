# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 09:41:24 2019
利用scikit-learn实现鸢尾花三分类问题
@author: Kylin
"""

import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
#注意：load_iris返回的iris对象是一个Bunch对象，与字典类似，包含键和值

X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

#利用X_train中的数据创建DataFrame
#利用iris_dataset.feature_names中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset['feature_names'])

#利用DataFrame创建散点图矩阵，按y_train着色。
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15),
                        marker='o', hist_kwds={'bins':20}, s=60,
                        alpha=0.8, cmap='viridis')

#实例化k近邻分类对象
knn = KNeighborsClassifier(n_neighbors=1)

#对训练集中的数据进行拟合
knn.fit(X_train, y_train)

#对测试集中的数据进行预测
y_pred = knn.predict(X_test)

#评估模型，输出预测精度
print("Test set score:{:.2f}".format(np.mean(y_pred == y_test)))


    