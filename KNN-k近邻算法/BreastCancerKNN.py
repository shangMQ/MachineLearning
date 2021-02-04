# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
利用KNN对乳腺癌数据进行拟合
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    # 1. 读取数据
    cancer = load_breast_cancer()
    # 2. 数据预处理， 使用stratify属性指定分层分配
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

    # 3. 利用不同的邻居数拟合模型
    training_accuracy = []
    test_accuracy = []
    neighbours_settings = range(1,11)

    for n_neighbour in neighbours_settings:
        clf = KNeighborsClassifier(n_neighbors= n_neighbour)
        clf.fit(X_train, y_train)
        # 查看拟合精度
        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))

    # 绘制精度
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False
    plt.plot(neighbours_settings, training_accuracy, 'g-o', label="training accuracy")
    plt.plot(neighbours_settings, test_accuracy, 'r-o', label="test accuracy")
    plt.xlabel("Number of neighbour")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
