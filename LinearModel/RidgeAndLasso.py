# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
测试两种带正则化的线性模型对特征的影响
L1:Lasso 特征选择
L2:Ridge 岭回归 降低每个特征对模型的影响
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import matplotlib as mpl


def main():
    mpl.rcParams[u'font.sans-serif'] = 'Times New Roman'
    mpl.rcParams[u'axes.unicode_minus'] = False

    # 1. 加载数据
    boston = load_boston()
    print("特征数量：",len(boston['feature_names']))
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
    print("训练集大小：", X_train.shape)

    # 2. 使用普通的线性回归
    linearReg = LinearRegression().fit(X_train, y_train)
    print("=======LinearRegression=========")
    print("线性回归得到的各个特征的相关权重系数：", linearReg.coef_)
    print("linearTrainScore : ", linearReg.score(X_train, y_train))
    print("linearTestScore : ", linearReg.score(X_test, y_test))

    # 3. 使用岭回归
    ridge1 = Ridge(alpha=0.1)
    ridge2 = Ridge(alpha=1)
    ridge3 = Ridge(alpha=10)
    ridge1.fit(X_train, y_train)
    ridge2.fit(X_train, y_train)
    ridge3.fit(X_train, y_train)
    plt.plot(linearReg.coef_, 's', color='blue', label="linear Regression")
    plt.plot(ridge1.coef_, '^', color='green', label="alpha=0.1")
    plt.plot(ridge2.coef_, 'v', color='orange', label="alpha=1")
    plt.plot(ridge3.coef_, 'o', color='red', label="alpha=10")
    plt.xlabel("coefficient index")
    plt.ylabel("value")
    plt.title("coefficient change under different alpha")
    plt.legend()
    plt.show()
    print("=======Ridge=========")
    print("ridge1TrainScore:", ridge1.score(X_train, y_train))
    print("ridge2TrainScore:", ridge2.score(X_train, y_train))
    print("ridge3TrainScore:", ridge3.score(X_train, y_train))
    print("ridge1TestScore:", ridge1.score(X_test, y_test))
    print("ridge2TestScore:", ridge2.score(X_test, y_test))
    print("ridge3TestScore:", ridge3.score(X_test, y_test))

    # 4. 使用lasso回归
    lasso1 = Lasso(alpha=0.1, max_iter=100)
    lasso2 = Lasso(alpha=1, max_iter=100)
    lasso1.fit(X_train, y_train)
    lasso2.fit(X_train, y_train)
    plt.plot(linearReg.coef_, 's', color='blue', label="linear Regression")
    plt.plot(ridge1.coef_, 'o', color='red', label="ridge:alpha=0.1")
    plt.plot(lasso1.coef_, '^', color='green', label="alpha=0.1")
    plt.plot(lasso2.coef_, 'v', color='orange', label="alpha=1")

    plt.xlabel("coefficient index")
    plt.ylabel("value")
    plt.title("coefficient change under different alpha")
    plt.legend()
    plt.show()
    print("=======Lasso=========")
    print("lasso1TrainScore:", lasso1.score(X_train, y_train))
    print("lasso2TrainScore:", lasso2.score(X_train, y_train))
    print("lasso1TestScore:", lasso1.score(X_test, y_test))
    print("lasso2TestScore:", lasso2.score(X_test, y_test))


if __name__ == "__main__":
    main()