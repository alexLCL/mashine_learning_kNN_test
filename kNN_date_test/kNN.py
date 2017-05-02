#-*- coding: UTF-8 -*-
from numpy import *  #科学计算模块
import operator            #运算符模块
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


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
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
	line = line.strip()
	listFromLine = line.split('\t')
	returnMat[index,:] = listFromLine[0:3]
	classLabelVector.append(listFromLine[-1]) #修改：去掉int()
	index += 1
    return returnMat,classLabelVector




def autoNorm(dataSet):
    minVals = dataSet.min(0) #参数0表示从列中选取最小值，并非当前行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  #生成矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals


# >>> import sys
# >>> sys.path.append("D:\python")
# >>> import kNN
# >>> from numpy import *
# >>> import matplotlib
# >>> import matplotlib.pyplot as plt
# >>> fig = plt.figure()
# >>> ax = fig.add_subplot(111)
# >>> mat,lab = kNN.file2matrix('D:\python\datingTestSet2.txt')
# >>> ax.scatter(mat[:,1], mat[:,2], 15.0*array(map(int,lab)),15.0*array(map(int,lab))) #重点修改本行
# >>> plt.show()


def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat , ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with : %s , the real naswer is: %s"%(classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]) : errorCount += 1.0
    print "the total error rate is : %f" %(errorCount/float(numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent palying video games?"))
    ffMiles = float(raw_input("ferquent flier miles earned per year ?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat , datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "you will probable like this person:",resultList[int(classifierResult) - 1] #str和int类型不能直接进行运运算，需要转换一下


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #获取目录下的内容
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))        #创建一个m行1024列的矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
