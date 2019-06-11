# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split   # cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + '正确率：\t', float(acc.sum()) / a.size)


if __name__ == "__main__":
    #1. 获取数据集
    data = np.loadtxt('wine.data', dtype=float, delimiter=',')
    #第一列数据为类标，之后的为特征值
    y, x = np.split(data, (1,), axis=1)
    # x = StandardScaler().fit_transform(x)
    #划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

    # Logistic回归（做对比时使用）
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    show_accuracy(y_hat, y_test, 'Logistic回归 ')

    # XGBoost
    y_train[y_train == 3] = 0 #类别为0、1、2
    y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    show_accuracy(y_hat, y_test, 'XGBoost ')