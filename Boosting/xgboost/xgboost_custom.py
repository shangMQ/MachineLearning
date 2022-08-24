# /usr/bin/python
# -*- encoding:utf-8 -*-
import os

import xgboost as xgb
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.datasets import make_blobs
from scipy.stats import ks_2samp

def gradient(predt, dtrain):
    """
    一阶导数
    """
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt+1)

def hessian(predt, dtrain):
    """
    二阶导数
    """
    y = dtrain.get_label()
    return ((- np.log1p(predt) + np.log1p(y) + 1) / np.power(predt+1, 2))

def squared_log(predt, dtrain):
    """
    定义目标函数
    """
    predt[predt<1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

def rmsle(predt, dtrain):
    """
    自定义评估指标
    """
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(acc)
    print(tip + '正确率：\t', float(acc.sum()) / a.size)

def ks(predt, dtrain):
    y = dtrain.get_label()
    return "ks", ks_2samp(predt[y==1], predt[y!=1]).statistic


if __name__ == '__main__':
    print(os.getcwd())
    # 1. 生成数据
    X, y = make_blobs(n_samples=10000, n_features=10, centers=2,
                      shuffle=True, random_state=0)

    # 划分数据集，训练集中有100个样本，测试集中有50个样本
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=50)

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 10, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

    results=dict()
    xgb.train(param,
              dtrain=data_train,
              num_boost_round=10,
              feval=ks,
              evals=[(data_train, 'dtrain'), (data_test, 'dtest')],
              evals_result=results)

    # bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    # y_hat = bst.predict(data_test)
    # show_accuracy(y_hat, y_test, 'XGBoost ')
