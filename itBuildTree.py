import numpy as np
import random
from math import log, floor

def getRandomForest(X,Y,K,classCount,M):

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
    bagSize = int(a/M)
        
    # Divide data randomly into M bags and create random tree for each
    for m in range(M):
        bag = randPerm[m*bagSize:min((m+1)*bagSize,a-1)]
        randomForest[m] = buildTree(X[bag,:],Y[bag],classCount,K)
        
    return randomForest

def treeClassify(A,N):
    # If not a leaf, test data against node's function
    if(N.Label == None):
        t = N.test(A)
        leftSplit = A[t == 0,:]
        rightSplit = A[t == 1,:]
        Yhat = np.zeros(A.shape[0])
        Yhat[t == 0] = treeClassify(leftSplit,N.leftChild)
        Yhat[t == 1] = treeClassify(rightSplit,N.rightChild)
        return Yhat

    # Else return the class of the leaf
    else:
        return N.Label

def buildTree(X,Y,classCount,k):
    n = Node(X,Y,k,classCount,np.ones([len(Y),1]))
    Tree = [n]
    leftToExpand = [n]
    while len(leftToExpand) > 0:
        # Split data with respect to first node in queue
        sortLeft = leftToExpand[0].test(X)
        sortRight = np.logical_not(sortLeft)
        leftClasses = Y[sortLeft]
        rightClasses = Y[sortRight]

        leftIsLeaf = False
        rightIsLeaf = False
        # Left child is a leaf, create leaf and don't enqueue
        for i in range(0,classCount):
            if(sum(abs(LeftClasses-i)) == 0 and not leftIsLeaf):
                Tree.append(Node(Label=i))
                leftIsLeaf = True
        # Left child isn't leaf, so enqueue node
        if(not leftIsLeaf):
            n = Node(X,Y,k,classCount,sortLeft)
            Tree.append(n)
            leftToExpand.append(n)
        # Right child is a leaf, create leaf and don't enqueue
        for i in range(0,classCount):
            if(sum(abs(RightClasses-i)) == 0 and not rightIsLeaf):
                Tree.append(Node(Label=i))
                rightIsLeaf = True
        # Right child isn't leaf, so enqueue node
        if(not rightIsLeaf):
            n = Node(X,Y,k,classCount,sortRight)
            Tree.append(n)
            leftToExpand.append(n)
        # Dequeue first node
        leftToExpand = leftToExpand[1:]

    return Tree


class Node(object):
    """ 
        X is the training parameters (n x m matrix)
        Y is the training labels (n x 1 matrix)
        K is the number of parameters considered at each node
        If Y is one class, Label is this class
        Intializes with the test function with highest information gain
    """
    def __init__(self, X = None, Y = None, k = None, classCount = None, currentSamples = None, Label = None):
        self.leftChild = None
        self.rightChild = None
        self.parent = None
        self.Label = Label
        self.currentSamples = currentSamples
        if(Label == None):
            self.Thresh, self.feature = findBestF(X,Y,k,classCount)

    def test(self,X):
        sortLabels = X[:,self.feature] < self.Thresh
        return np.dot(sortLabels.astype('int'),currentSamples.astype('int'))

def findBestF(X,Y,k,classCount):

    samples,parameters = np.shape(X)
    # J = random.sample(range(parameters),k)
    # J = range(parameters)
    A = np.sort(X[:,:200],0)

    Thresholds = (A[:-1,:] + A[1:,:])/2

    numOfThresh = 100
    s = np.linspace(0,samples-2,numOfThresh,dtype=int)
    Thresholds = Thresholds[s,:]


    minH = float("infinity")
    bestF = []
    bestDiv = []

    for i in range(numOfThresh):
        for j in range(k):
            sortLabels = np.array(X[:,J[j]] < Thresholds[i,j]).astype('int')

            HA = 0
            HB = 0 

            nA = sum(sortLabels)
            nB = samples - nA

            # Calculate entropy
            for y in range(0,classCount):
                classTruth = np.array(Y == y).astype('int')
                totalInClass = sum(classTruth)
                pA = np.dot(classTruth,sortLabels)
                pB = totalInClass - pA
                if(pA == 0 or nA == 0):
                    HA += 0
                else:
                    HA += -pA/nA*log(float(pA)/nA)/log(2)
                if(pB == 0 or nB == 0):
                    HB += 0
                else:
                    HB += -pB/nB*log(float(pB)/nB)/log(2)

            H = (HA*nA + HB*nB)/samples

            if(H < minH):
                bestT = Thresholds[i,j]
                bestF = J[j]
                minH = H
    
    return bestT, bestF

def DFS(N,X):
    """ Counts numer of nodes """
    if(N.leftChild == None):
        if(N.rightChild == None):
            return 1
        else:
            n1 = DFS(N.rightChild,X)
    elif(N.rightChild == None):
        if(N.leftChild == None):
            return 1
        else:
            n2 = DFS(N.rightChild)
    elif(N.leftChild != None and N.rightChild != None):
        print('Left: ' + str(sum(N.test(X))) + '. Right: ' + str(X.shape()[0]-sum(N.test(X))))
        n2 = DFS(N.leftChild,X)
        n1 = DFS(N.rightChild,X)
    return n1 + n2 + 1

