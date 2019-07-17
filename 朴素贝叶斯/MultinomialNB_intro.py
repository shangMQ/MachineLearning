#!/usr/bin/python
#测试朴素贝叶斯模型的效果
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB


if __name__ == "__main__":
    #1. 加载数据
    np.random.seed(0)
    M = 20#样本数
    N = 5#特征数
    #特征向量X
    x = np.random.randint(2, size=(M, N))     # [low, high)
    x = np.array(list(set([tuple(t) for t in x])))#去掉重复的特征

    M = len(x)#更新样本数
    y = np.arange(M)
    
    print('样本个数：%d，特征数目：%d' % x.shape)
    print('样本：\n', x)
    
    #2.构建模型
    mnb = MultinomialNB(alpha=1)    # 动手：换成GaussianNB()试试预测结果？
    gnb = GaussianNB()
    #3. 拟合模型
    mnb.fit(x, y)
    gnb.fit(x, y)
    
    #4. 预测结果
    y_hat = mnb.predict(x)
    y_hat2 = gnb.predict(x)
    print('mnb预测类别：', y_hat)
    print('gnb预测类别：', y_hat2)
    
    
    #两种模型准确率计算方法
    print('mnb准确率：%.2f%%' % (100*np.mean(y_hat == y)))
    print('mnb系统得分：', mnb.score(x, y))
    print('gnb准确率：%.2f%%' % (100*np.mean(y_hat2 == y)))
    print('gnb系统得分：', gnb.score(x, y))
    
    # from sklearn import metrics
    # print metrics.accuracy_score(y, y_hat)
    
    #输出分类错误的特征向量
    print("分类错误样本详情：")
    err = (y_hat != y)
    for i, e in enumerate(err):
        if e:
            print("第{:}个样本数据{:}被认为与{:}一个类别".format(y[i], x[i], x[y_hat[i]]))
