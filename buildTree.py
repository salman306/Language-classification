import numpy as np
import random
from math import log, floor

def getRandomForest(X,Y,K,classCount,M)

    """ 
        X is the training parameters (n x m matrix)
        Y is the training labels (n x 1 matrix)
        K is the number of parameters considered at each node
        classCount is the number of unique classes in Y
        M is the number of subsets we divide our data into
    """
    a,b = np.shape(X)

    # Get random permutation 
    randPerm = np.random.permutation(range(a))

    # Get size of each bag
    bagSize = floor(a/M)

    # Divide data randomly into M bags and create random tree for each
    for m in range(M):
        bag = randPerm[m*bagSize:min((m+1)*bagSize,a-1)]
        randomForest[m] = buildTree(X[bag,:],Y[bag],classCount,K)

    return randomForest

def treeClassify(A,N):
    # If not a leaf, test data against node's function
    if(N.Label == None):
        t = N.test(A)
        if(t == 0):
            return treeClassify(A,N.leftChild)
        else:
            return treeClassify(A,N.rightChild)
    # Else return the class of the leaf
    else:
        return N.Label

def buildTree(X,Y,classCount,k):
    # If Y contains a single class create a leaf node
    for i in range(0,classCount):
        if(sum(abs(Y-i)) == 0):
            return Node(Label=i)
    # Otherwise create a node and two children for the split data
    n = Node(X,Y,k,classCount) 
    n.leftChild = buildTree(X[n.div],Y[n.div],classCount,k)
    n.rightChild = buildTree(X[~n.div],Y[~n.div],classCount,k)
    n.leftChild.parent = n
    n.rightChild.parent = n
    return n


class Node(object):
    """ 
        X is the training parameters (n x m matrix)
        Y is the training labels (n x 1 matrix)
        K is the number of parameters considered at each node
        If Y is one class, Label is this class
        Intializes with the test function with highest information gain
    """
    def __init__(self, X = None, Y = None, k = None, classCount = None, Label = None):
        self.leftChild = None
        self.rightChild = None
        self.parent = None
        self.Label = Label
        if(Label == None):
            self.Thresh, self.feature, self.div = findBestF(X,Y,k,classCount)
        else:
            self.F = None
            self.div = None

    def test(self,X):
        sortLabels = X[:,self.feature] < self.Thresh
        return sortLabels

def findBestF(X,Y,k,classCount):

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
    """ Counts numer of nodes """
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

