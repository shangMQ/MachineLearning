#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
#利用四个特征来构建决策树

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False

   #利用pandas加载数据集
    path = 'iris.data'  # 数据文件路径
    data = pd.read_csv(path, names=["sepal-length", "sepal-width",
                                     "petal-length", "petal-width", "label"])
    #为了便于处理数据，将label标记为int型数据
    data["label"] = pd.Categorical(data["label"]).codes
    
    
    #x_prime为特征数组，y为类标数组
    x = np.array(data[["sepal-length", "sepal-width", "petal-length", "petal-width"]])
    y = np.array(data["label"])
    
    # 决策树学习
    clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    dt_clf = clf.fit(x, y)
    y_hat = dt_clf.predict(x)
    c = np.count_nonzero(y_hat == y)    # 统计预测正确的个数
    print('\t预测正确数目：', c)
    print('\t准确率: %.2f%%' % (100 * float(c) / float(len(y))))

