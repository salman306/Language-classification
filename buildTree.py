from NodeTest import Test
from NodeTest import evalTest  
import numpy as np

class Node(object):
    """docstring for Node"""
    def __init__(self, X, Label = None):
        self.leftChild = None
        self.rightChild = None
        self.F, self.div = findBestF(X)
        self.Label = Label

    def test(X):
        sortLabels = self.F.splitData(X)
        return sortLabels

def buildTree(X,Y,classCount):
    for i in range(0,classCount):
        if(sum(Y-i) == 0):
            return Node(X,Label=i)
    n = Node(X)
    n.leftChild = buildTree(X[n.div],Y[n.div],classCount)
    n.rightChild = buildTree(X[~n.div],Y[~n.div],classCount)
    return n

def findBestF(X):

    classCount = 5

    a,b = np.shape(X)
    A = np.sort(X,0)

    Thresholds = (A[:-1,:] + A[1:,:])/2

    minH = 1
    bestF = []
    bestDiv = []

    for i in range(0,a):
        for j in range(0,b):
            F = Test(j,-1,Thresholds[i,j])

            labels, H = evalTest(X,Y,F,classCount)
            if(H < minH):
                bestF = F
                minH = H
                bestDiv = labels

    return F, bestDiv