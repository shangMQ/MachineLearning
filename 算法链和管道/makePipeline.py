# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:30:39 2019
make_pipeline可以自动创建管道并根据每个步骤所属的类别对其自动命名
@author: Kylin
"""
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

#1. 标准语法
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C = 100))])

#2. 缩写
pipe_short = make_pipeline(PCA(n_components=2), SVC(C=100))

#3. 可以通过steps属性查看步骤名称
print(pipe_short.steps)

#4. 加载数据
cancer = load_breast_cancer()

#5. 利用管道拟合数据集
pipe_short.fit(cancer.data, cancer.target)

#6. named_steps属性，它是一个字典，可以将步骤名称映射为估计器。
components = pipe_short.named_steps["pca"].components_
print("components shape:", components.shape)
