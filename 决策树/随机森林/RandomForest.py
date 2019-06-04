# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:09:00 2019
利用sklearn包中的RandomForestClassifier构建随机森林
@author: Kylin
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
y_hat = forest.predict(X_test)
score = np.mean(y_hat == y_test)
print("准确率：", score)
