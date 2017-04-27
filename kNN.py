#-*- coding: UTF-8 -*-
from numpy import *  #科学计算模块
import operator            #运算符模块
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                       #读取dataSet的第一维度的 长度，也就是获取它的行数，示例为4行
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #把inX重复dataSetSize次，后面的1保证是4行而不是1行，减去dataSet为了坐标相减求距离
    sqDiffMat = diffMat ** 2                                   #坐标相减的平方
    sqDistances = sqDiffMat.sum(axis = 1)           #axis=1代表列相加
    distances = sqDistances**0.5                            #开根号
    sortedDistIndicies = distances.argsort()           #argsort() 从小到大排序，并返回坐标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1     #get是取字典里的元素，
                                                                                                            # 如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），
                                                                                                            #这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1

        sortedClassCount = sorted(classCount.items(), key =operator.itemgetter(1), reverse = True)  #key=operator.itemgetter(1)的意思是按照字典里的第一个排序，
                                                                                                                                                                        # {A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector=[]
    enumLabelVector=[]
    fr = open(filename)
    index =0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[3])
        #产生一个类别向量，但是记录的是整型而非string
        if cmp(listFromLine[3],'didntLike')==0:
           enumLabelVector.append(1)
        elif  cmp(listFromLine[3],'smallDoses')==0:
            enumLabelVector.append(2)
        elif cmp(listFromLine[3],'largeDoses')==0:
            enumLabelVector.append(3)
        index+=1
    fr.close()
    return returnMat,classLabelVector,enumLabelVector

mat,vector1,vector2 = file2matrix('datingTestSet.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(mat[:,1],mat[:,2],15.0*array(vector2),15.0*array(vector2))
plt.show()