# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:25:07 2019
嵌套交叉验证
外层循环遍历数据划分为训练集和测试集的所有划分，对于每种划分都运行一次网格搜索。
然后对于每种外层划分，利用最佳参数设置计算得到测试集score。
@author: Kylin
"""
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import warnings

warnings.filterwarnings("ignore")

#1. 加载数据
iris = load_iris()

#2. 用字典指定要搜索的参数，考虑到不同核对应的参数不同，将它们放在一个列表字典中
#键是要调节的参数名称，值是尝试的参数设置
param_grid = [{"kernel":["rbf"],"C":[0.001, 0.01, 0.1, 1, 10, 100], "gamma":[0.001, 0.01, 0.1, 1, 10, 100]}, 
              {"kernel":["linear"],"C":[0.001, 0.01, 0.1, 1, 10, 100]}]

#3.对于每种划分都运行一次网格搜索
print("————————直接将GridSearchCV()实例化传入cross_val_score的方式———————————")
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), iris.data, iris.target, cv=5)
print("交叉验证scores:", scores)
print("交叉验证score的均值:", scores.mean())