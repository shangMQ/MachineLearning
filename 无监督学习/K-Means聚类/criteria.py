# !/usr/bin/python
"""聚类的衡量指标：同一性、完整性
    同一性：一个簇只包含一个类别的样本
    完整性：同类别的样本归到同一个簇中（这个计算我个人还是不太懂）
    但是，同一性和完整性的值都很大时，往往聚类的效果较好。
"""
# -*- coding:utf-8 -*-

from sklearn import metrics


if __name__ == "__main__":
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 1, 2, 2]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    v2 = 2 * c * h / (c + h)
    v = metrics.v_measure_score(y, y_hat)
    print('V-Measure：', v2, v)

    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 2, 3, 3]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    v = metrics.v_measure_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    print('V-Measure：', v)

    # 允许不同值
    print
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [1, 1, 1, 0, 0, 0]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    v = metrics.v_measure_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    print('V-Measure：', v)

    y = [0, 0, 1, 1]
    y_hat = [0, 1, 0, 1]
    ari = metrics.adjusted_rand_score(y, y_hat)
    print(ari)

    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 1, 2, 2]
    ari = metrics.adjusted_rand_score(y, y_hat)
    print(ari)
