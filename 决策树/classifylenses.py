# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:43:21 2019
使用决策树预测隐形眼镜类型
@author: Kylin
"""
import tree
import matplotlib.pyplot as plt
import treePlotter

#1. 获得数据
fr = open("lenses.txt")

#2. 解析数据
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenseLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
fr.close()

#3. 训练算法：使用createTree()函数
lensesTree = tree.createTree(lenses,lenseLabels)
print(lensesTree)

#4. 绘制树形图
treePlotter.createPlot(lensesTree)

