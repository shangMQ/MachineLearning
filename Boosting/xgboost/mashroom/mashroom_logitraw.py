# /usr/bin/python
# -*- encoding:utf-8 -*-

"""
使用xgboost预测蘑菇是否有毒
主要介绍xgboost的基本用法
使用logitraw分类
"""

import xgboost as xgb

#1. 读取数据
#data_train，data_test是XGboost的DMatrix自定义类型
data_train = xgb.DMatrix('agaricus_train.txt')
data_test = xgb.DMatrix('agaricus_test.txt')

#2. 设置参数
#max_depth:树的最大深度；eta：衰减因子；silent：是否不显示树的生成过程；objective：
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logitraw'} # logitraw
watchlist = [(data_test, 'eval'), (data_train, 'train')]
n_round = 3 #迭代n次，生成n棵树
bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)

# 计算错误率
y_hat = bst.predict(data_test) #预测测试集的类标 
y = data_test.get_label() #获取测试集的实际类标
error = sum(y != (y_hat > 0)) #这里和logistic不同
error_rate = float(error) / len(y_hat)
print('-'*5, '分类函数使用logitraw', '-'*5)
print('样本总数：\t', len(y_hat))
print('错误数目：\t%4d' % error)
print('错误率：\t%.5f%%' % ( 100 * error_rate))
