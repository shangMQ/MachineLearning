# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:29:23 2019
利用cancer数据集，尝试对logistic回归进行L2调参
@author: Kylin
"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#加载数据集
cancer = load_breast_cancer()

#划分数据训练集和测试集，stratify是为了保持split前类的分布
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    stratify=cancer.target, random_state=42)

#默认C=1
#训练数据
logreg = LogisticRegression().fit(X_train, y_train)
#输出精度值
print("使用默认的L2级正则化系数即C=1时：")
print("训练集精度:{:.3f}".format(logreg.score(X_train, y_train)))
print("测试集精度:{:.3f}".format(logreg.score(X_test, y_test)))


#C=100
#训练数据
logreg2 = LogisticRegression(C=100).fit(X_train, y_train)
#输出精度值
print("使用L2级正则化系数即C=100时：")
print("训练集精度:{:.3f}".format(logreg2.score(X_train, y_train)))
print("测试集精度:{:.3f}".format(logreg2.score(X_test, y_test)))

#C=0.01
#训练数据
logreg3 = LogisticRegression(C=0.01).fit(X_train, y_train)
#输出精度值
print("使用默认的L2级正则化系数即C=0.01时：")
print("训练集精度:{:.3f}".format(logreg3.score(X_train, y_train)))
print("测试集精度:{:.3f}".format(logreg3.score(X_test, y_test)))

#查看正则化参数C取三个不同的值时模型学到的系数
plt.rcParams['font.sans-serif']=['simhei'] #设置显示中文title的字体
fig = plt.figure("L2级正则化效果")
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg2.coef_.T, '^', label="C=100")
plt.plot(logreg3.coef_.T, 'v', label="C=0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.title("不同L2级正则化系数对参数维度的影响")
plt.xlabel("系数索引")
plt.ylabel("系数维度")
plt.legend()

#使用L1级正则化
fig2 = plt.figure("L1级正则化效果")

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C = C, penalty='l1').fit(X_train, y_train)
    print("使用L1级正则化系数即C={:.3f}时".format(C))
    print("训练集精度:{:.3f}".format(lr_l1.score(X_train, y_train)))
    print("测试集精度:{:.3f}".format(lr_l1.score(X_test, y_test)))
    label = "C={:.3f}".format(C)
    plt.plot(lr_l1.coef_.T, 'v', label=label)
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.title("不同L1级正则化系数对参数维度的影响")
plt.xlabel("系数索引")
plt.ylabel("系数维度")
plt.legend()

    


