import operator
from math import log
import matplotlib.pyplot as plt


# calcSHannonEnt 这个函数是用来计算信息增熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] #查看数据里的label 一个一个的读取
        if currentLabel not in labelCounts.keys(): #这个if循环 是用来统计这个字典里某个键出现的次数 也就是计算yes 和 no的个数
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts: #当labelCounts 为0的时候就计算 no对应的 信息增熵 为-3/5*log2（-3/5）大概是这样  然后计算出yes的加起来输出
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt


#hyj这个函数是用来看一个集合里出现某个东西的个数的另一种写法 与上面 if currentLabel not in ..... 这个程序功能差不多
'''
def hyj(dataSet):
    numlen2 = len(dataSet)
    hyj = {}
    for hyj1 in dataSet:
        c1 = hyj1[-1]
        hyj[c1] = hyj.get(c1, 0) + 1
        
'''


#本程序的数据
def creatDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


my_dataset, my_labels = creatDataSet()
#hyj(my_dataset)
#my_dataset[0][-1] = 'maybe'
hyj4 = calcShannonEnt(my_dataset)
print(hyj4)


#splitDataSet 这个函数是用来分离数据 dataSet是用来分离的数据 axis是依据第几列 value是条件
#比如说 按照第一列 0来分数据 axis=0 value=0  dataSet得到的结果就是 [1,'no'] [1,'no']也就是数据里最后两个
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet: #一列一列看数据
        if featVec[axis] == value: #如果选择条件等于featVec中的值就会执行下面程序
            reducedFeatVec = featVec[:axis] #把featVec中对应数据的前面一部分拿出来
            reducedFeatVec.extend(featVec[axis+1:])#再取对应数据后面一部分的数据  比如[1,3,4]如果关键词取到了3 上面一个程序就是取1 这一步程序就是取4 然后组合成了[1,4]
            retDataSet.append(reducedFeatVec) # 这一步是讲[1,4]这样得到的多个列表组合起来
    return retDataSet
#.extend 和 .append 的区别
# a = [1,2,3]
# b = [4,5,6]
# a.extend(b) = [1,2,3,4,5,6]
# a.append(b) = [1,2,3,[4,5,6]]


hyj5 = splitDataSet(my_dataset, 0, 1)
print(hyj5)


# 选择用第几个特征来当作合适的决策树的分支 被选择后的特征再下一次选分支的时候是不参与的
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1  #数据最后一个是label 减去一个就是特征个数
    baseEntropy = calcShannonEnt(dataSet) #这一步是计算一下数学公式里的Info（D）
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures): #这个大循环的次数与特征值个数相关 特征值几个就循环几次
        featList = [example[i] for example in dataSet] #这个就是将列表里 同一列的数提取出来 比如 i=0 就提出了[1，1，1，0，0] 这个
        uniqueVals = set(featList) #python里函数 功能将相同的数保留一个 比如上面的就保留了只有[1,0]
        newEntropy = 0.0
        for value in uniqueVals: #这个循环跟每个特征值对应的状态相关 比如特征值1 有 是和否两种状态 这里就是两次 三种就是三次
            subDataSet = splitDataSet(dataSet, i, value) #这一行和下面一行程序是用来算出 某一特征值 某个状态占多少 比如 特征值1 里面有4个1 6个0 那么这个功能就是用来算4个1占了4/10
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet) #计算比如上面1对应的信息增益 如先算出推到公式InfoA（D） 计算过程
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain: #比较每个特征值的大小 保留大的和第几列
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature



hyj6 = chooseBestFeatureToSplit(my_dataset)
print(hyj6)



def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]





#输出决策树
#这个函数是一个套娃很多很多循环建议多看几遍
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):   #.count是用来看字符串在里面出现了几次
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择哪个特征来当分支
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

myTree = createTree(my_dataset,my_labels)
print(myTree)