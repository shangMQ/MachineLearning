# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:33:56 2019
比较PCA、KMeans在重建图像中的效果
对于Kmeans重建就是在训练集中找到最近的簇中心
@author: Kylin
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt

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
    
    return faces, face_shape

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

def showPic(X_test, X_reconstructed_pca, X_reconstructed_km, kmeans, pca, imageshape):
    fig, axes = plt.subplots(2, 5, figsize=(8,8), subplot_kw={"xticks":(), "yticks":()})
    fig.suptitle("Extracted Components")
    
    for ax, comp_pca, comp_km in zip(axes.T, kmeans.cluster_centers_, pca.components_):
        ax[0].imshow(comp_pca.reshape(imageshape))
        ax[1].imshow(comp_km.reshape(imageshape))
    
    axes[0,0].set_ylabel("PCA")
    axes[1,0].set_ylabel("KMeans")
    
    fig2, axes = plt.subplots(3, 5, subplot_kw={"xticks":(), "yticks":()}, figsize=(8,8))
    fig2.suptitle("Reconstruction")
    for ax, origin, rec_pca, rec_kmeans in zip(axes.T, X_test, X_reconstructed_pca, X_reconstructed_km):
        ax[0].imshow(origin.reshape(imageshape))
        ax[1].imshow(rec_pca.reshape(imageshape))
        ax[2].imshow(rec_kmeans.reshape(imageshape))
        
    axes[0,0].set_ylabel("Origin")
    axes[1,0].set_ylabel("PCA")
    axes[2,0].set_ylabel("KMeans")
    
if __name__ == "__main__":
    #1. 加载数据
    faces, face_shape = loadData()
    X_faces, y_faces = eliminateInclination(faces)
    X_train, X_test, y_train, y_test = train_test_split(X_faces,y_faces,stratify=y_faces, random_state=0)
    
    #2. 拟合不同的模型
    pca = PCA(n_components=100, random_state=0)
    pca.fit(X_train)
    kmeans = KMeans(n_clusters=100, random_state=0)
    kmeans.fit(X_train)
    
    #3. 利用不同模型重建后的测试数据集
    X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
    X_reconstructed_km = kmeans.cluster_centers_[kmeans.predict(X_test)]
    
    #4. 绘制原测试集图像和重建后的图像
    showPic(X_test, X_reconstructed_pca, X_reconstructed_km, kmeans, pca, face_shape)