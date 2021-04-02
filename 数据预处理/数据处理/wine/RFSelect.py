# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
基于随机森林的特征选择
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def loadData():
    """
    加载数据
    :return: 数据特征，类别
    """
    df_wine = pd.read_csv("wine.data", header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    return X, y, df_wine.columns[1:]


def main():
    # 1. 加载数据
    X, y, columns = loadData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 2. 构建随机森林分类器
    rf = RandomForestClassifier(n_estimators=400, random_state=0)
    rf.fit(X_train, y_train)

    # 3. 查看特征重要性
    importance = rf.feature_importances_
    print(importance)
    indices = np.argsort(importance)[::-1]
    print(indices)
    for f in range(X_train.shape[1]):
        col = indices[f]
        print("(%02d) %s %f" % (f+1, columns[col], importance[col]))

    # 4. 特征可视化
    plt.bar(range(X_train.shape[1]), importance[indices], color='lightblue', align='center')
    plt.xticks(range(X_train.shape[1]), columns[indices], rotation=90)
    plt.title("Feature Importance")
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()