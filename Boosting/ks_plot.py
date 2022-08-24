# /usr/bin/python
# -*- encoding:utf-8 -*-
"""
ks and roc plot with xgboost and save model
"""

import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split   # cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + '正确率：\t', float(acc.sum()) / a.size)

def cal_ks(pred, y):
    fpr, tpr, threshold = roc_curve(y, pred)
    ks_value = max(abs(fpr - tpr))

    # 记录ks对应的阈值索引
    ks_index = np.argmax(abs(tpr - fpr))
    ks_threshold = threshold[ks_index]
    ks_tpr = tpr[ks_index]
    ks_fpr = fpr[ks_index]
    print(f"The threshold of ks = {ks_threshold}")

    # 结果可视化
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # roc曲线
    plt.subplot(1,2,1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC")

    # ks曲线
    plt.subplot(1,2,2)
    plt.plot(threshold, tpr, 'b-', label="tpr")
    plt.plot(threshold, fpr, 'g-', label="fpr")
    # ks对应的tpr和fpr
    plt.scatter(ks_threshold, ks_fpr, color='r', edgecolors='k', s=30)
    plt.scatter(ks_threshold, ks_tpr, color='r', edgecolors='k', s=30)
    plt.plot([ks_threshold, ks_threshold], [ks_fpr, ks_tpr], 'r--')
    # 绘制中间部分
    plt.fill_between(threshold, fpr, tpr, facecolor='pink', alpha=0.6)

    plt.legend()
    plt.xlabel("threshold")
    plt.ylabel("tpr/fpr")
    plt.title("KS")

    plt.show()

    return ks_value


if __name__ == "__main__":
    #1. 获取数据集
    data = np.loadtxt('wine.data', dtype=float, delimiter=',')
    #第一列数据为类标，之后的为特征值
    y, x = np.split(data, (1,), axis=1)
    # x = StandardScaler().fit_transform(x)
    #划分数据集
    arg_index = np.argwhere(y!=3)[:,0]
    y = y[arg_index]
    x = x[arg_index]
    # print(x.shape)
    # print(y.shape)
    y[y == 2] = 0

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

    # XGBoost
    # y_train[y_train == 3] = 0 #类别为0、1、2
    # y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    # 二分类
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    # 多分类
    # param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class':3, 'eval_metric':'mlogloss'}
    results = dict()
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)

    # 模型保存
    bst.save_model("xgboost_wine.json")

    # 得到的结果是一个概率
    pred_test = bst.predict(data_test)

    # 计算ks
    ks = cal_ks(pred_test, y_test)
    print(f"ks value = {ks}")

    # 模型加载
    # 方法一
    reload_xgb = xgb.Booster()
    reload_xgb.load_model("xgboost_wine.json")
    pred = reload_xgb.predict(data_train)
    print(pred)

    # 方法二
    clf = XGBClassifier()
    booster = xgb.Booster()
    booster.load_model('xgboost_wine.json')
    clf._Booster = booster
    clf.predict(x_train)

