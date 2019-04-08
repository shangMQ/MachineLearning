# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:00:08 2019

@author: Kylin
"""

import matplotlib.pyplot as plt
import tree

#定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
totalW = -1
totalD = -1

#绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制节点函数
    输入参数：节点文本，节点中心坐标，起点坐标，节点类型
    """
    #annotate添加文本注解
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def getNumLeafs(myTree):
    """
    获取叶节点的数目
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #根节点字符串
    secondDict = myTree[firstStr] #根节点对应的值是另一个字典或一个叶节点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #如果节点的类型是dict，说明他们不是叶节点
            numLeafs += getNumLeafs(secondDict[key])#递归获取这个字典中的叶节点数
        else:
            numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    """
    判断树的层数，计算遍历过程中遇到判断节点的个数
    该函数的终止条件是叶子节点，一旦到达叶子节点，则从递归调用中返回，
    并将计算树深度变量加1
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #如果是判断节点，则当前深度计算如下：
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    """
    计算父节点和子节点的中间位置，
    并此处添加父子节点间填充文本信息
    """
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    #if the first key tells you what feat was split on
    """
    绘制树形图
    """
    #首先，计算树的宽和高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    #获得父节点信息
    firstStr = list(myTree.keys())[0]
    
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    #获取父节点键所对应的值（可能是另一个字典，也可能是叶子）
    secondDict = myTree[firstStr]
    #按比例减少yOff值
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #如果元素是决策节点，就递归这个节点
            plotTree(secondDict[key],cntrPt,str(key))
        else:   
            #如果是一个叶子节点，则在图形上画出叶子节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    #利用全局变量来存储树的宽度和深度，以便可以计算树节点的摆放位置
    #树的宽度用于计算放置判断节点的摆放位置，将它放在素哟有叶子节点的中间
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    #全局变量，xOff和yOff用于追踪已经绘制的节点位置，以及放置下一个节点的恰当位置
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

if __name__ == "__main__":
    myDat, labels = tree.createDataSet()
    Tree = tree.createTree(myDat, labels)
    createPlot(Tree)