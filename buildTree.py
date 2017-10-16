import numpy as np
import random
from math import log

class Node(object):
    """docstring for Node"""
    def __init__(self, X = None, Y = None, k = None, Label = None):
        self.leftChild = None
        self.rightChild = None
        self.parent = None
        self.Label = Label
        if(Label == None):
            self.Thresh, self.feature, self.div = findBestF(X,Y,k)
        else:
            self.F = None
            self.div = None

    def test(self,X):
        sortLabels = X[:,self.feature] < self.Thresh
        return sortLabels

def buildTree(X,Y,classCount,k):
    for i in range(0,classCount):
        if(sum(abs(Y-i)) == 0):
            return Node(Label=i)
    n = Node(X,Y,k) 
    n.leftChild = buildTree(X[n.div],Y[n.div],classCount,k)
    n.rightChild = buildTree(X[~n.div],Y[~n.div],classCount,k)
    n.leftChild.parent = n
    n.rightChild.parent = n
    return n

def findBestF(X,Y,k):

    classCount = 5

    a,b = np.shape(X)
    J = random.sample(range(b),k)
    A = np.sort(X[:,J],0)

    Thresholds = (A[:-1,:] + A[1:,:])/2

    minH = float("infinity")
    bestF = []
    bestDiv = []

    for i in range(a-1):
        for j in range(k):
            sortLabels = X[:,J[j]] < Thresholds[i,j]

            HA = 0
            HB = 0

            nA = sum(sortLabels)
            nB = a - nA

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

            H = (HA*nA + HB*nB)/a

            if(H < minH):
                bestT = Thresholds[i,j]
                bestF = J[j]
                minH = H
                bestDiv = sortLabels
    return bestT, bestF, bestDiv

def DFS(N):
    if(N.leftChild == None):
        if(N.rightChild == None):
            return 1
        else:
            n1 = DFS(N.rightChild)
    elif(N.rightChild == None):
        if(N.leftChild == None):
            return 1
        else:
            n2 = DFS(N.rightChild)
    elif(N.leftChild != None and N.rightChild != None):
        n1 = DFS(N.rightChild)
        n2 = DFS(N.leftChild)
    return n1 + n2 + 1

def treeClassify(A,N):
    if(N.Label == None):
        t = N.test(A)
        if(t == 0):
            return treeClassify(A,N.leftChild)
        else:
            return treeClassify(A,N.rightChild)
    else:
        return N.Label