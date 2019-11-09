# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:59:20 2019

@author: Kylin
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit

def plotdata(X,y):
    class1 = X[(y == 0)]
    class2 = X[(y == 1)]
    plt.scatter(class1[:,0], class1[:,1], color="red", label="class1")
    plt.scatter(class2[:,0], class2[:,1], color="green", label="class2")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.title("Moon dataset")

#1. 生成moons数据
X,y = make_moons(n_samples=10000, noise=0.4)

#2. 可视化数据集
plotdata(X,y)

#3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. 构建参数列表字典
params = {'max_leaf_nodes': list(range(2, 100)),
          'min_samples_split': [2, 3, 4]}

#5. 使用GridSearch和3折交叉验证找到最佳超参数
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
print("最佳超参数模型：", grid_search_cv.best_estimator_)
print("最佳超参数：",grid_search_cv.best_params_)

#5. 生成训练集的1,000个子集，每个子集包含100个随机选择的实例
n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=0.2, random_state=42)

#注意：这里返回的是下标
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

#5. 利用之前找到的最佳超参数拟合数据，之后进行测试
from sklearn.base import clone

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    
    pred_score = tree.score(X_test, y_test)
    accuracy_scores.append(pred_score)

print("测试均值：", np.mean(accuracy_scores))

#6. 对于每个测试集实例，生成1,000个决策树的预测，并仅保留最频繁的预测（您可以为此使用SciPy的mode（）函数）
#这样就可以对测试集进行多数表决的预测。
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
    
from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))