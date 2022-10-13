import time

import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics, datasets
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import random as random
from time import strftime, localtime
import datetime


def load_data(filename, id_col=None, y_col=None):
    from pathlib import Path
    if not Path(filename).exists():
        raise ValueError(f'{filename} dose not exist.')

    df = pd.read_csv(filename)
    df = df.fillna(0)

    if id_col:
        df = df.set_index(id_col)

    if y_col:
        if y_col not in df.columns:
            raise ValueError(f'{y_col} not in schema!')

        x_cols = [c for c in df.columns if c != y_col]
        x_df = df[x_cols]
        y_df = df[[y_col]]
    else:
        x_df = df
        y_df = None

    return x_df, y_df

def evaluate(y, y_pred):
    """
    评估方法，主要评估auc和ks指标
    :param y:
    :param y_pred:
    :return:
    """
    y_pred = list(y_pred[:, 1])
    fpr, tpr, thresholds = roc_curve(y, y_pred)

    # AUC
    auc = metrics.auc(fpr, tpr)
    # KS
    ks = max(tpr - fpr)

    return auc, ks

if __name__ == "__main__":
    start = time.time()

    # 加载数据
    guest_file = "../data/breast_hetero_guest.csv"
    host_file = "../data/breast_hetero_host.csv"

    # 加载数据，先加载guest的，id_col指定id列的列名，y_col指定y列的列名
    x_guest, y_guest = load_data(guest_file, id_col="id", y_col="y")

    # 加载host的，id_col指定id列的列名，y_col指定y列的列名（当前host没有y）
    x_host, _ = load_data(host_file, id_col="id")
    print("====Initial data=====")
    print(f"guest shape = {x_guest.shape}")
    print(f"host shape = {x_host.shape}")

    # 数据集合并
    X = pd.merge(x_guest, x_host, on="id")
    id_col = pd.DataFrame(X.index, columns=["id"])
    X = X.values
    y_guest = pd.merge(id_col, y_guest, on="id")
    y_guest.set_index("id", inplace=True)

    # 训练集测试集划分，train_size控制训练集的比例
    X_train, X_test, y_train, y_test = train_test_split(X, y_guest, train_size=0.9, random_state=0)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    print("======After splited========")
    print(f"X_train shape = {X_train.shape}")
    print(f"y_train shape = {y_train.shape}")
    print(f"X_test shape = {X_test.shape}")
    print(f"y_test shape = {y_test.shape}")

    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 最大迭代次数
    max_iter = 100
    # batch size
    mini_batch = 200

    cls = linear_model.SGDClassifier(loss="log_loss", penalty="l2", alpha=0.1, max_iter=100, learning_rate='constant', eta0=0.1, fit_intercept=False) # SGDClassifier with loss="log"

    classes = np.array([0, 1])
    auc_list = []
    ks_list = []
    
    for i in range(max_iter):
        batch_index = np.random.choice(X_train.shape[0], mini_batch, replace=False)
        X_batch = X_train[batch_index]
        y_batch = y_train[batch_index]
        cls.partial_fit(X_batch, y_batch, classes=classes)

    y_train_pred = cls.predict_proba(X_train)
    y_test_pred = cls.predict_proba(X_test)

    # evaluate
    print("=====Evaluate======")
    train_auc, train_ks = evaluate(y_train, y_train_pred)
    print(f'train_auc = {train_auc}, train_ks = {train_ks}')
    test_auc, test_ks = evaluate(y_test, y_test_pred)
    print(f'test_auc = {test_auc}, test_ks = {test_ks}')
