from numpy import *
import operator
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

#定义一组训练数据    一组标签A、B函数
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group, labels = createDataSet()

print(type(group))
print(group)
print(labels)
# 这里的shape在二维矩阵里 shape[0]就是行 shape[1]就是列
hyj = group.shape[0]
print((hyj))



#KNN算法
#通过判断输入数据与数据集之间的距离来判断输入数据是哪一类的
#两点距离公式这种


#在这里有个问题 classCount.items（）这个地方在python3中是items 书上写的是iteritems

#tile()函数是numpy中的如果需要用from numpy import *
#tile()函数  tile((1, 2), 3) 结果就是array([1,2,1,2,1,2])
#当是个矩阵时候 比如 a=array([[1, 2],[3, 4]])两行两列 那么tile(a, 2) 就是就是这样的a = array([[1,2,1,2],[3,4,3,4]])变成了两行四列
#title(a,(2,2)) 最后一个2还是和上面一样 多了两列 前面一个2是多了两行 就变成了a = array([[1,2,1,2],[3,4,3,4],[1,2,1,2],[3,4,3,4]])
#title(a,(行,列))

#.argsort()是用来排序的从小到大 输出的是原来数据对应的索引比如(-2,5,-1) 正常排序是(-2,-1,5)  argsort()函数输出的是(0,2,1)这样的索引


# classify0这个函数是用来实现KNN算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]    #求出数据集行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #主要是求出输入数据与数据集所有点的差 例如：x-x1 y-y1的差这种
    sqDiffMat = diffMat**2                          #这里是输入数据与数据集差的平方 （x-x1）**2平方
    sqDistances = sqDiffMat.sum(axis=1)             #求解(x-x1)**2+(y-y1)**2
    distances = sqDistances**0.5                    #开根号求出距离
    sortedDistIndicies = distances.argsort()        #排列大小
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #这个函数是用来计算出一个字典里相同元素的个数 比如出现了几次
        #reverse True 降序 False 升序
        #key 在sorted里面一般是用来排序时候 用来的判据 比如这里operator.itemgetter(1)就是用 2 1这个大小来排序 itemgetter(0)就是B和A
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 下面这个链接是用来讲classCount.get这个用法的
# https://blog.csdn.net/aaa9692/article/details/115469795?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-115469795-blog-79743428.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-115469795-blog-79743428.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=1

#验证KNN算法
h = classify0([0, 0], group, labels, 3)
print(h)



'''
print('**************')
a = array([[[1, 2, 3],[4, 5, 6]]])
print(a)
print(shape(a))

'''

#file2matrix()这个函数处理文本文件 将其分为数据集 和 标签（3 2 1 这种）



#.strip()这个函数的功能是删除开始和结尾处的空格（当括号里什么都没写的时候）
#例如 输入为s = ' 1, 3, 4 ' s.strip()输出就是s = '1, 3, 4'
#如果 括号里添加了东西 不如strip(',') 就可以删除前和尾上的,
#例如 s = ',1, 3, 4,' 输出就是s = '1, 3, 4'

#.split()这个函数是用来拆分字符串的
#比如 line = line.strip()这里的输出是'4090\t8\t0.32\t3\n'
#split('\t') 就可以依据\t这个来拆分这个数据 拆分为 4090，8，0.32，3 了 想拆分依据什么拆分里面就写啥

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numbersOfLines = len(arrayOLines)
    returnMat = zeros((numbersOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

datingDataMat,datingLabels = file2matrix('code/Ch02/datingTestSet2.txt')
print(datingDataMat)
print(datingLabels[0:20])


'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
'''

#autoNorm 是将数据归一化（大数据基本上都要用到这个过程）
#在矩阵里min[0]列最小 min[1]行最小 max同理
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges  = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals




hyjj,hyj1,hyj2 = autoNorm(datingDataMat)


#datingClassTest()用来测试数据集的准确性
#\作用是续行的作用
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('code/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d"\
              %(classifierResult, datingLabels[i]))
        if(classifierResult !=datingLabels[i]):errorCount +=1.0
    print("the total error rate is:%f" %(errorCount/float(numTestVecs)))

datingClassTest()



'''
def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('code/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult-1])

classifyPerson()

'''



'''
#下面这个函数是用来转换数据中喜欢 非常喜欢 不喜欢为 2 3 1的

def preprocess_data(data_2):
    data_3 = data_2.copy()
    data_3 = data_3.replace(['didntLike', 'smallDoses', 'largeDoses'], value=[1, 2, 3])
    return data_3


#替换好感度为1 2 3 （下面这个方法会把第一行的数据移除了）
df2 = pd.read_table('code/Ch02/datingTestSet.txt')
hyj = preprocess_data(df2)
print(hyj)

'''



#下面是手写字体KNN算法识别
#readline()函数他是一行一行读数据 readlines()是全部都读取出来
def img2vector(filename):
    retunVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            retunVect[0, 32*i+j] = int(lineStr[j])
    return retunVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('code/Ch02/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('code/Ch02/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('code/Ch02/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('code/Ch02/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with : %d, the real answer is: %d" % (classifierResult, classNumStr))
        if(classifierResult !=classNumStr):
            errorCount += 1.0

    print("\nthe total number of errors is:%d" % errorCount)
    print("\nthe total error rate is: %f "%(errorCount/float(mTest)))


handwritingClassTest()