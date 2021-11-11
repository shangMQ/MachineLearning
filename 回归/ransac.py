# -*- coding:utf-8 -*-
"""
ransac 随机抽样一致性算法
· 可消除异常值对回归算法的影响
· 首先假设数据具有某种特性（目的），为了达到目的，适当割舍一些现有的数据。
· 虽然降低了数据集中异常点的潜在影响，但无法确定该方法对未知数据的预测是否存在正面影响

:Author: Kylin
:Last Modified by: Kylin.smq@qq.com
"""
import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib as mpl


def lr_predict(X, y):
    """
    使用线性回归拟合
    """
    lr = LinearRegression()
    lr.fit(X, y)
    lr_w = lr.coef_
    lr_intercept = lr.intercept_
    y_lr = lr_w * X + lr_intercept

    return y_lr


def ransac_predict(X, y):
    """
    使用随机抽样一致性检验
    """
    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                             residual_threshold=5, random_state=0)
    ransac.fit(X, y)

    # 内点集合
    inliner_mask = ransac.inlier_mask_
    # 外点集合
    outliner_mask = np.logical_not(inliner_mask)
    # 预测结果
    line_y_ransac = ransac.estimator_.coef_ * X + ransac.estimator_.intercept_

    return inliner_mask, outliner_mask, line_y_ransac


def main():
    # 1.生成数据
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 1.5 * X + np.random.randn(100).reshape(-1, 1)

    # 手动插入几个异常值
    y[40] = 15
    y[42] = 18
    y[45] = 22
    y[48] = 25
    y[55] = 27
    y[60] = 30
    X_outlier = np.array([X[40], X[42], X[45], X[48], X[55], X[60]])
    y_outlier = np.array([y[40], y[42], y[45], y[48], y[55], y[60]])

    # 直接使用线性回归拟合
    y_lr = lr_predict(X, y)

    # 使用随机抽样一致性模型
    inliner_mask, outliner_mask, y_ransac = ransac_predict(X, y)

    # 查看效果
    # 设置中文字体
    mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
    mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure("LinearRegression VS Ransac")
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    ax = fig.add_subplot(1, 3, 1)
    plt.scatter(X, y, s=25, label="normal")
    plt.scatter(X_outlier, y_outlier, color='red', edgecolors='k', s=40, label='abnormal')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("initial Data")
    plt.legend()

    ax = fig.add_subplot(1, 3, 2)
    plt.plot(X, y_lr, color='k', linewidth=4)
    plt.scatter(X, y, s=25, label="normal")
    plt.scatter(X_outlier, y_outlier, color='red', edgecolors='k', s=40, label='abnormal')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()

    ax = fig.add_subplot(1, 3, 3)
    plt.plot(X, y_ransac, color='k', linewidth=4)
    plt.scatter(X[inliner_mask], y[inliner_mask], s=25, label="normal")
    plt.scatter(X[outliner_mask], y[outliner_mask], color='red', edgecolors='k', s=40, label='abnormal')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Random Sample Consensus Regression")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

