# coding:utf-8
import numpy as np
"""
    KMeans使用案例
"""
import sklearn.datasets as ds
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1 加载数据
x, y = ds.make_blobs(400, n_features=2, centers=4, random_state=2018)

# 2 拟合模型
model = KMeans(n_clusters=4, init='k-means++')
model.fit(x)
# 使用cluster_centers_可以查看质心
print("聚类质心：")
print(model.cluster_centers_)
y_pred = model.predict(x)

# 3 查看结果，评估算法
print('y = ', y[:30])
print('y_Pred = ', y_pred[:30])
# 均一性，一个簇只包含一个类别的样本
# 若n个样本被分为了n个簇，那么这个聚类的均一性一定是最好的，但是完整性确是最差的.
print('homogeneity_score = ', homogeneity_score(y, y_pred))
# 完整性，同类样本被归类到相同簇中
# 若使得聚类的簇一直减小，直到剩下一个簇，那么这个聚类的完整性是最好的，但是，均一性是最差的。
print('completeness_score = ', completeness_score(y, y_pred))
# 均一性和完整性的加权平均
print('v_measure_score = ', v_measure_score(y, y_pred))

print('adjusted_mutual_info_score = ', adjusted_mutual_info_score(y, y_pred))
print('adjusted_rand_score = ', adjusted_rand_score(y, y_pred))
# 轮廓系数
print('silhouette_score = ', silhouette_score(x, y_pred))

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(x[:, 0], x[:, 1], 'r.', ms=3)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Initial Data")

plt.subplot(122)
plt.scatter(x[:, 0], x[:, 1], c=y_pred, marker='.', cmap=mpl.colors.ListedColormap(list('rgbm')))
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='d', color='y', s=100)
plt.title("Clusters")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout(2)
plt.suptitle("KMeans Tests")
plt.show()
