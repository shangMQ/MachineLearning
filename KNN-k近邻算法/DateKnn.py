# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:35:06 2019
在约会网站上使用k近邻算法
----来自《Machine Learning in Action》
@author: Kylin
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import operator  # 导入运算符模块


def file2matrix(filename):
    """
         处理输入格式问题函数，输入为文件名，输出为训练样本矩阵和类标签向量
    """
    # 准备数据，使用Python解析文本文件
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 得到文件中的行数
    returnMat = np.zeros((numberOfLines, 3))  # 准备训练矩阵
    classLabelVector = []   # 准备标签向量
    fr = open(filename)  # 重新加载一次，因为默认readlines会读到末尾
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, np.array(classLabelVector)


def plotData(dataSet, labels):
    """
    数据可视化
    :param dataSet:
    :param labels:
    :return:
    """
    mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制图例
    Astart = np.where(labels == 1)[0][0]
    Bstart = np.where(labels == 2)[0][0]
    Cstart = np.where(labels == 3)[0][0]
    AData = dataSet[Astart]
    BData = dataSet[Bstart]
    CData = dataSet[Cstart]

    type1 = plt.scatter(AData[0], AData[1], s=30, edgecolors='k', c='#501964')
    type2 = plt.scatter(BData[0], BData[1], s=30, edgecolors='k', c='#4E9E9D')
    type3 = plt.scatter(CData[0], CData[1], s=30, edgecolors='k', c='yellow')
    plt.legend((type1, type2, type3), ('Dislike', 'Ordinary', 'harming'), loc=2)

    # 绘制图
    plt.scatter(dataSet[:, 0], dataSet[:, 1], s=30, c=labels, edgecolors='k')
    plt.xlabel("Air Time(mile)")
    plt.ylabel("Play(%)")
    plt.title("Visualize the information of men")
    plt.show()


def autoNorm(dataSet):
    """
    归一化数据
    处理不同取值范围的特征值时，通常采用将数值归一化，例如将取值范围处理为0到1或者-1到1之间。
    自动将数字特征值转化为0到1之间
    :param dataSet:
    :return:
    """
    # dataSet.min(0)中的参数0使得函数可以从列中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    测试算法
    :return:
    """
    hoRatio = 0.10   # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')   # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]

    # 测试集numTestVecs是样本总数m的10%
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        # numTestVecs利用normMat的前i行数据进行测试，之后的用于训练集，k=3
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs: m], 3)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)


def classify(inX, dataSet, labels, k):
    """
        参数简介：
        inX: 用于分类的输入向量
        dataSet: 输入的训练样本集
        labels: 标签向量
        k: 选择最近邻居的数目
    """
    dataSetSize = dataSet.shape[0]

    # 1. 计算已知类别数据集中的点与当前点之间的距离
    # np.tile()将用于分类的输入向量纵向赋值为dataSetSize个, 在与每个dataSet作差
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    # 为了避免出现负数，平方，因为距离均为正值
    sqDiffMat = diffMat ** 2
    # 计算和(按照行求和)
    sqDistances = sqDiffMat.sum(axis=1)
    # 再开方，计算出与每个训练集中的欧式距离
    distances = sqDistances ** 0.5

    # 2. 按照距离递增顺序排序，返回一个下标数组
    sortedDistIndicies = distances.argsort()

    # 3. 选取与当前点距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 4. 确定前k个点所在类别出现的频率,operator.itemgetter()用于获取对象哪些维的数据
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)

    # 5. 返回前k个点出现频率最高的类别作为当前点的预测类别。
    return sortedClassCount[0][0]


def classifyPerson():
    """
    算法使用
    :return:
    """
    resultList = ['不喜欢', '一般', '很喜欢']
    print("-"*5, "Kylin家园交友网站", "-"*5)
    print("请输入男嘉宾的相关信息:")
    percentTats = float(input("玩游戏所消耗的时间百分比？"))
    ffMiles = float(input("每年的飞行里程有多少？"))
    iceCream = float(input("每周消耗的冰激凌公升数？"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("你对他感觉：", resultList[classifierResult-1])


if __name__ == '__main__':
    # 1. 收集数据,放在文件datingTestSet2.txt,每个样本数据占一行共3个特征，共1000行
    file = "datingTestSet2.txt"
    datingDataMat, datingLabels = file2matrix(file)

    # 2. 分析数据，利用matplotlib绘制散点图
    plotData(datingDataMat, datingLabels)

    # 3. 数据标准化
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    plotData(normDataSet, datingLabels)

    # 4. 使用模型
    classifyPerson()
