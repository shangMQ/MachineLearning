# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
添加多项式特征对波士顿房价数据集的拟合结果
在线性模型上添加多项式特征往往有利于提高训练精度，而随机森林则不用添加多项式特征
"""
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def main():
    # 1. 读取数据
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

    # 2. 数据预处理
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. 使用多项式交互特征
    poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
    X_train_poly = poly.transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    # 查看数据集大小变化
    print("=====Initial Data=======")
    print("train data shape:", X_train_scaled.shape)
    print("test data shape:", X_test_scaled.shape)

    print("=====Poly Data=======")
    print("train data shape:", X_train_poly.shape)
    print("test data shape:", X_test_poly.shape)
    print("扩展后特征名：", poly.get_feature_names())

    # 4. 使用线性模型
    # 4.1 使用原始特征
    ridge1 = Ridge().fit(X_train_scaled, y_train)
    ridge1_train = ridge1.score(X_train_scaled, y_train)
    ridge1_test = ridge1.score(X_test_scaled, y_test)
    print("*******Linear Model************")
    print("========Initial Score=======")
    print("Train Score : ", ridge1_train)
    print("Test Score : ", ridge1_test)

    # 4.2 使用多项式交互特征
    ridge2 = Ridge().fit(X_train_poly, y_train)
    ridge2_train = ridge2.score(X_train_poly, y_train)
    ridge2_test = ridge2.score(X_test_poly, y_test)
    print("========Poly Score=======")
    print("Train Score : ", ridge2_train)
    print("Test Score : ", ridge2_test)

    # 5. 使用随即森林模型
    rf1 = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
    rf1_train = rf1.score(X_train_scaled, y_train)
    rf1_test = rf1.score(X_test_scaled, y_test)
    print("*******Random forest************")
    print("=======Initial Data========")
    print("Train Score : ", rf1_train)
    print("Test Score : ", rf1_test)

    rf2 = RandomForestRegressor(n_estimators=50).fit(X_train_poly, y_train)
    rf2_train = rf2.score(X_train_poly, y_train)
    rf2_test = rf2.score(X_test_poly, y_test)
    print("=======Poly Data========")
    print("Train Score : ", rf2_train)
    print("Test Score : ", rf2_test)


if __name__ == "__main__":
    main()