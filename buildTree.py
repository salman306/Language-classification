import numpy as np
from math import log

class Node(object):
    """docstring for Node"""
    def __init__(self, X = None, Y = None, Label = None):
        self.leftChild = None
        self.rightChild = None
        self.parent = None
        self.Label = Label
        if(Label == None):
            self.F, self.div = findBestF(X,Y)
        else:
            self.F = None
            self.div = None

    def test(X):
        sortLabels = self.F.splitData(X)
        return sortLabels

class Test(object):
    """docstring for Test
    column is the feature on which the test is performed.
    test is either -2,-1,0,1,2 which is translated into <,<=,=,>=,> respectively
    critValue is the value on which the data is split as per the specific test"""
    def __init__(self, column, test, critValue):
        self.column = column
        self.test = test
        self.critValue = critValue

    def splitData(self,X):
        if(self.test == -2):
            sortLabels = X[:,self.column] < self.critValue 
        elif(self.test == -1):
            sortLabels = X[:,self.column] <= self.critValue 
        elif(self.test == 0):
            sortLabels = X[:,self.column] = self.critValue 
        elif(self.test == 1):
            sortLabels = X[:,self.column] >= self.critValue 
        elif(self.test == 2):
            sortLabels = X[:,self.column] > self.critValue 

        return sortLabels


def buildTree(X,Y,classCount):
    for i in range(0,classCount):
        if(sum(abs(Y-i)) == 0):
            return Node(Label=i)
    n = Node(X,Y)
    n.leftChild = buildTree(X[n.div],Y[n.div],classCount)
    n.rightChild = buildTree(X[~n.div],Y[~n.div],classCount)
    n.leftChild.parent = n
    n.rightChild.parent = n
    return n

def findBestF(X,Y):

    classCount = 5

    a,b = np.shape(X)
    A = np.sort(X,0)

    Thresholds = (A[:-1,:] + A[1:,:])/2

    minH = float("infinity")
    bestF = []
    bestDiv = []

    for i in range(0,a-1):
        for j in range(0,b):
            F = Test(j,-1,Thresholds[i,j])

            labels, H = evalTest(X,Y,F,classCount)
            if(H < minH):
                bestF = F
                minH = H
                bestDiv = labels

    return F, bestDiv

def evalTest(X,Y,Test,classCount):

    sampleCount = len(Y)

    # sortLabels are false if sample failed test and true o.w.
    sortLabels = Test.splitData(X)

    HA = 0
    HB = 0

    nA = sum(sortLabels)
    nB = sampleCount - nA

    # Calculate entropy
    for y in range(0,classCount):
        pA = sum([a == y and b for a, b in zip(Y,sortLabels)])
        pB = sum([a == y and not b for a, b in zip(Y,sortLabels)])
        if(pA == 0):
            HA += 0
        else:
            HA += -pA/nA*log(pA/nA)/log(2)
        if(pB == 0):
            HB += 0
        else:
            HB += -pB/nB*log(pB/nB)/log(2)

    # Calculate conditional entropy for passing/failing test
    CondEntropy = (HA*nA + HB*nB)/sampleCount

    return [sortLabels,CondEntropy]