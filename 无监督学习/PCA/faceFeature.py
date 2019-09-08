# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:03:35 2019
对于面部识别的特征提取
@author: Kylin
"""
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def loadData():
    """
    加载数据
    """
    faces = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    
    #查看每张图片的大小
    face_shape = faces.images[0].shape
    print("每张图片的像素大小:", face_shape)
    
    #查看图片总个数
    print("faces.images.shape:", faces.images.shape)
    
    #查看类别总数
    print("图片来自{}位名人".format(len(faces.target_names)))
    
    return faces

def showFacesPic(faces):
    """
    查看前十张图片
    """
    fig, axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(), 'yticks':()})
    
    for target, image, ax in zip(faces.target, faces.images, axes.ravel()):
        ax.imshow(image)
        ax.set_title(faces.target_names[target])

def showPCAPic(pca, faces):
    """
    显示利用PCA处理后的数据图像
    """
    image_shape = faces.images[0].shape
    
    fig, axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(), 'yticks':()})
    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap="viridis")
        ax.set_title("{}.component".format((i+1)))
        print("component shape:", component.shape)

def celebrityCount(faces):
    """
    查看名人目标图像的个数
    """
    print("The number of pictures with each celebrity:")
    counts = np.bincount(faces.target)
    for i, (count, name) in enumerate(zip(counts, faces.target_names)):
        print("({},{})".format(name, count))

def eliminateInclination(faces):
    """
    降低数据倾斜，每个人最多只用50张图像
    """
    mask = np.zeros(faces.target.shape,  dtype=np.bool)
    for target in np.unique(faces.target):
        mask[np.where(faces.target == target)[0][:50]] = 1
    
    X_faces = faces.data[mask]
    y_faces = faces.target[mask]
    
    #为了得到更好的数据稳定性，将灰度值缩放到0到1之间，而不是0到255之间
    X_faces = X_faces/255
    return X_faces, y_faces


if __name__ == "__main__":
    #1. 加载数据
    faces = loadData()
    
    #2. 查看前十张脸部图片
    #showFacesPic(faces)
    
    #3. 查看每个名人的图片数目
    celebrityCount(faces)
    #可以发现数据偏斜，Bush的图片数就很多。
    
    #4. 此时考虑降低数据偏斜，否则特征提取会被数量多的目标影响
    X_faces, y_faces = eliminateInclination(faces)
    
    #5. 将数据集分为测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(X_faces, y_faces, stratify=y_faces, random_state=0)
    
    #6. 使用一个邻居构建K近邻分类器
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print("-----单邻居KNN分类器------")
    print("train set score：{:.2f}".format(knn.score(X_train, y_train)))
    print("test set score：{:.2f}".format(knn.score(X_test, y_test)))
    
    #7. 对训练数据拟合PCA对象(提取前100个主成分)
    pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("X_train_pca shape:", X_train_pca.shape)
    print("pca.components_shape:{}".format(pca.components_.shape))
    
    #8. 对经过PCA处理的数据应用K近邻分类器
    print("-----PCA处理后的数据构建单邻居KNN分类器------")
    knn.fit(X_train_pca, y_train)
    print("train set score：{:.2f}".format(knn.score(X_train_pca, y_train)))
    print("test set score：{:.2f}".format(knn.score(X_test_pca, y_test)))

    #9. 查看经过pca处理后的数据图像
    showPCAPic(pca, faces)
    
    #10. 利用pca的前两个主要成分，将数据集中的所有人脸在散点图中可视化
    fig = plt.figure("Two Principal Components")
    plt.scatter(X_train_pca[:,0], X_train_pca[:,1], y_train)
    plt.xlabel("The first component")
    plt.ylabel("The second component")
    plt.title("Two Principal Components")
