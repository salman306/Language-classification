import numpy as np
import random
from math import log, floor
from multiprocessing.dummy import Pool as ThreadPool

def mutli_run_wrapper(args):
    return buildTree(*args)

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

    pool = ThreadPool(M)
        
    # Divide data randomly into M bags and create random tree for each
    args = []
    for m in range(M):
        bag = randPerm[m*bagSize:min((m+1)*bagSize,a-1)]
        args.append((X[bag,:],Y[bag],classCount,K))

    randomForest = pool.map(buildTree,args)
        
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
    # If Y contains a single class create a leaf node
    for i in range(0,classCount):
        if(sum(abs(Y-i)) == 0):
            return Node(Label=i)
    # Otherwise create a node and two children for the split data
    if(len(Y) < 100):
        classDist = []
        for i in range(0,classCount):
            classDist.append(sum(Y == i))
        return Node(Label=np.argmax(classDist))
    n = Node(X,Y,k,classCount) 
    div = n.test(X)
    n.leftChild = buildTree(X[div],Y[div],classCount,k)
    sortRight = np.logical_not(div)
    n.rightChild = buildTree(X[sortRight],Y[sortRight],classCount,k)
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
            self.Thresh, self.feature = findBestF(X,Y,k,classCount)

    def test(self,X):
        sortLabels = X[:,self.feature] < self.Thresh
        return sortLabels

def findBestF(X,Y,k,classCount):

    samples,parameters = np.shape(X)
    # J = random.sample(range(parameters),k)
    J = range(parameters)
    A = np.sort(X[:,J],0)

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

