# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:27:23 2019
城市共享自行车的使用情况
数据信息：
    提供的数据为2年内按小时做的自行车租赁数据，
    其中训练集由每个月的前19天组成，测试集由20号之后的时间组成。
@author: 尚梦琦
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

"""
----------------
1. 查看数据信息
----------------
"""
data = pd.read_csv("kaggle_bike_competition_train.csv", header=0, error_bad_lines=False)

#将数据的日期域切分成日期和时间两部分
datetime = pd.DatetimeIndex(data["datetime"])
data["date"] = datetime.date
data["time"] = datetime.time

#时间的最小粒度是小时，可将小时作为一个字段
data["hour"] = pd.to_datetime(data.time, format="%H:%M:%S")
data["hour"] = pd.Index(data["hour"]).hour

#周末和工作日出去的人的数量应该有所不同，设定字段dayofweek表示一周中的第几天
data["dayofweek"] = pd.DatetimeIndex(data.date).dayofweek

#设定dateDays表示距离第一天开始租车的时间
data["dateDays"] = (data.date - data.date[0]).astype("timedelta64[D]")

#统计未注册用户的租赁情况
byday = data.groupby("dayofweek")
byday["casual"].sum().reset_index()

#统计注册用户的租赁情况
byday["registered"].sum().reset_index()

#对周末的数据单独考虑
data["Saturday"] = 0
data.Saturday[data.dayofweek==5] = 1
data["Sunday"] = 0
data.Sunday[data.dayofweek==6] = 1

#将数据中的原始时间字段剔除
dataRel = data.drop(['datetime', 'count','date','time','dayofweek'], axis=1)

"""
----------------
2. 特征处理
----------------
"""
#将连续值的属性放入一个dict中
featureConCols = ["temp", "atemp", "humidity", "windspeed", "dateDays", "hour"]
dataFeatureCon = dataRel[featureConCols]
dataFeatureCon = dataFeatureCon.fillna('NA') #填充空白数据
X_dictCon = dataFeatureCon.T.to_dict().values()

#把离散值的属性放到另外一个dict中
featureCatCols = ['season','holiday','workingday','weather','Saturday', 'Sunday']
dataFeatureCat = dataRel[featureCatCols]
dataFeatureCat = dataFeatureCat.fillna('NA') #填充空白数据
X_dictCat = dataFeatureCat.T.to_dict().values() 

#向量化特征
vec = DictVectorizer(sparse=False)
X_vec_cat = vec.fit_transform(X_dictCat)
X_vec_con = vec.fit_transform(X_dictCon)

#标准化连续值特征，使得特征符合标准正态分布
scaler = preprocessing.StandardScaler().fit(X_vec_con)
X_vec_con = scaler.transform(X_vec_con)

#类别特征利用one-hot编码
encoder = preprocessing.OneHotEncoder()
encoder.fit(X_vec_cat)
X_vec_cat = encoder.transform(X_vec_cat).toarray()

#将离散型和连续型的特征拼接到一起,X_vec的前六列是标准化后的连续值特征，后面是编码后的离散值特征
X_vec = np.concatenate((X_vec_con, X_vec_cat), axis=1)

#对结果值也进行处理
Y_vec_reg = dataRel["registered"].values.astype(float)
Y_vec_cas = dataRel["casual"].values.astype(float)
