import numpy as np
import operator


def classfier(intX, dataSet, labels, k):
    n = dataSet.shape[0]
    data = np.tile(intX, (n, 1)) - dataSet
    datasquare = data ** 2
    datasum = datasquare.sum(axis=1)
    datasqrt = datasum ** 0.5
    indexs = datasqrt.argsort()
    classcount = {}

    for i in range(k):
        #print(classcount)
        label = labels[indexs[i]]
        classcount[label] = classcount.get(label, 0) + 1
   

    res = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return res[0][0]


def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels






def file2matrix(filename):
    f = open(filename)
    lines = f.readlines()
    #print(lines)
    n = len(lines)
    matrix = np.zeros((n, 3))

    index = 0
    labels = []
    counts = {'largeDoses': 3, 'smallDoses': 2, 'didntLike':1}
    for item in lines:
        item = item.strip()
        a = item.split('\t')
        matrix[index, :] = a[0:3]
        if a[-1].isdigit():
            labels.append(int(a[-1]))
        else:
            labels.append(counts.get(a[-1]))
        index = index + 1

    return matrix, labels



def autoNorm(dataSet):
    n = dataSet.shape[0]
    a = dataSet - np.tile(dataSet.min(0), (n, 1))
    b = np.tile(dataSet.max(0) - dataSet.min(0))
    return a / b

group, labels = createDataSet()
pre = classfier([0, 0], group, labels, 3)
matrix, labels = file2matrix('datingTestSet.txt')
