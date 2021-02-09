# -*- coding:utf-8 -*-
"""
关于汽车销售的一个逻辑回归问题
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    # 相关设置
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)

    # 1. 读取数据
    data = pd.read_csv('car.data', header=None)
    print("Initial Data : \n", data.head())
    # 原始数据没有列名，默认从0开始
    # 设置合适的列名
    n_columns = len(data.columns)
    columns = ['buy', 'maintain', 'doors', 'persons', 'boot', 'safety', 'accept']
    new_columns = dict(list(zip(np.arange(n_columns), columns)))
    data.rename(columns=new_columns, inplace=True)
    print("New Data : \n", data.head(10))

    # 2. 数据预处理
    # 2.1 由于数据中包含一些类别特征，使用one-hot编码
    x = pd.DataFrame()
    for col in columns[:-1]:
        t = pd.get_dummies(data[col], prefix=str(col))
        # t = t.rename(columns=lambda x: col+'_'+str(x))
        x = pd.concat((x, t), axis=1)
    print("One hot processing:")
    print(x.head(10))
    print("data shape : ", x.shape)
    # 将标签用编码标记
    y = np.array(pd.Categorical(data['accept']).codes)
    print("label types : ", len(np.unique(y)))
    print("Accept Data:\n", data['accept'])
    print("Code of Accept:\n", y)

    # 3. 划分数据集，注意按照类别分层抽样
    x, x_test, y, y_test = train_test_split(x, y, stratify=y, test_size=0.3)

    # 3.1 利用逻辑回归
    # clf = LogisticRegressionCV(Cs=np.logspace(-3, 4, 8), cv=5)

    # 3.2 利用随机森林
    clf = RandomForestClassifier(n_estimators=50, max_depth=7)
    clf.fit(x, y)
    # print(clf.C_)
    y_hat = clf.predict(x)
    print('训练集精确度：', metrics.accuracy_score(y, y_hat))
    y_test_hat = clf.predict(x_test)
    print('测试集精确度：', metrics.accuracy_score(y_test, y_test_hat))
    n_class = len(np.unique(y))
    print("class types : ", n_class)

    if n_class > 2:
        # 由于有四种类型，对类别进行onehot编码
        y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
        # 预测给定测试集数据，在每种类别的可能性
        y_test_one_hot_hat = clf.predict_proba(x_test)
        fpr, tpr, _ = metrics.roc_curve(y_test_one_hot.ravel(), y_test_one_hot_hat.ravel())
        print('Micro AUC:\t', metrics.auc(fpr, tpr))
        auc = metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='micro')
        print('Micro AUC(System):\t', auc)
        auc = metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='macro')
        print('Macro AUC:\t', auc)
    else:
        fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), y_test_hat.ravel())
        print('AUC:\t', metrics.auc(fpr, tpr))
        auc = metrics.roc_auc_score(y_test, y_test_hat)
        print('AUC(System):\t', auc)

    # 绘制ROC曲线图
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False
    plt.figure(figsize=(8, 7), facecolor='w')
    plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)
    plt.legend(loc='lower right', fontsize=14)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.grid(b=True, ls=':')
    plt.title('ROC And AUC', fontsize=16)
    plt.show()
