# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
PCA特征提取方法的底层操作
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
    X, y, label = loadData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 2. 标准化c
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)

    # 3. 计算协方差矩阵
    cov_mat = np.cov(X_train_std.T)

    # 4. 计算协方差矩阵的特征值和特征向量，
    # 协方差的特征向量代表了主成分，
    # 对应的特征值大小决定了特征向量的重要性
    eigen_vals, eigen_vects = np.linalg.eig(cov_mat)
    print("特征值大小：\n", eigen_vals.shape)
    print("特征值：\n", eigen_vals)
    print("特征向量大小：\n", eigen_vects.shape)
    # print("特征向量：\n", eigen_vects)

    # 4. 对特征值排序，特征值决定了特征的重要性
    # 4.1 绘制特征值的方差贡献率图像
    # 特征值的方差贡献率：当前特征值与所有特征值之和的比值
    total = sum(eigen_vals)
    var_exp = [(i / total) for i in sorted(eigen_vals, reverse=True)]
    print("每个特征的贡献率：\n", var_exp)
    cum_var_exp = np.cumsum(var_exp)
    print("累积方差贡献：\n", cum_var_exp)

    plt.bar(range(1, 14), var_exp, alpha=0.6, align='center', edgecolor='k', label='individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")
    plt.title("Explained variance ratio change under different principal component")
    plt.legend()
    plt.show()

    # 4.2 特征值降序排列
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vects[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)

    # 4.3 选择特征值最大的特征向量（假设k=2）构造映射矩阵
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print("映射矩阵1：\n", w)

    # 5. 特征转换
    X_train_pca = X_train_std.dot(w)

    # 6. 利用散点图可视化
    colors = ['r', 'g', 'b']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Principal Component Show")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()