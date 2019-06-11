# /usr/bin/python
# -*- encoding:utf-8 -*-

"""
自定义损失函数的梯度和二阶导
"""
import xgboost as xgb
import numpy as np



def log_reg(y_hat, y):
    #自定义损失函数
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label() #梯度
    h = p * (1.0-p) #应该用到二阶导数信息(但是这个是logistic函数的一阶导数啊)
    return g, h


def error_rate(y_hat, y):
    #定义错误率
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == "__main__":
    #1. 读取数据
    #data_train，data_test是XGboost的DMatrix自定义类型
    data_train = xgb.DMatrix('agaricus_train.txt')
    data_test = xgb.DMatrix('agaricus_test.txt')

    #2. 设置参数
    #max_depth:树的最大深度；eta：衰减因子；silent：是否不显示树的生成过程；objective：
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'} # logistic
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 3 #迭代n次，生成n棵树
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)
