import numpy as np
import random
from math import log, floor
from multiprocessing.dummy import Pool
import os

STD = 1


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
        
    pool = Pool(2)

    # Divide data randomly into M bags and create random tree for each
    args = []
    for m in range(M):
        bag = randPerm[m*bagSize:min((m+1)*bagSize,a-1)]
        args.append((X[bag,:],Y[bag],classCount,K))

    randomForest = pool.map(mutli_run_wrapper,args)

    return randomForest

def forestClassify(X,F,classCount):
    yhat = []
    R = np.zeros([X.shape[0],len(F)])
    for i in range(len(F)):
        R[:,i] = treeClassify(X,F[i])
    # 
    for i in range(R.shape[0]):
        classDist = []
        for j in range(classCount):
            classDist.append(sum(R[i,:] == j))
        yhat.append(np.argmax(classDist))
    # 
    return yhat

def treeClassify(A,N):
    # If not a leaf, test data against node's function
    if(N.Label == None):
        t = N.sortTest(A)
        leftSplit = A[t == 0,:]
        rightSplit = A[t == 1,:]
        Yhat = np.zeros(A.shape[0])
        Yhat[t == 0] = treeClassify(leftSplit,N.leftChild)
        Yhat[t == 1] = treeClassify(rightSplit,N.rightChild)
        return Yhat

    # Else return the class of the leaf
    else:
        return N.Label

def buildTree(X,Y,classCount,K=150):    
    global STD
    STD = np.std(X,0)
    Tree = Node(X,Y,K,classCount,np.ones([len(Y)]).astype('bool'))
    # Breadth first order
    leftToExpand = [Tree]
    while len(leftToExpand) > 0:
        print(str(len(leftToExpand)) + ' (PID ' + str(os.getpid()) + ')')
        N = leftToExpand[0]
        # Split data with respect to first node in queue
        sortLeft,sortRight = leftToExpand[0].test(X)
        leftClasses = Y[sortLeft]
        rightClasses = Y[sortRight]

        # If best funcion doesn't split well, make leaf
        if(sum(sortRight) == 0):
            classDist = []
            for i in range(0,classCount):
                    classDist.append(sum(leftClasses == i))
            N.Label = np.argmax(classDist)
            leftToExpand = leftToExpand[1:]
            continue
        if(sum(sortLeft) == 0):
            classDist = []
            for i in range(0,classCount):
                    classDist.append(sum(rightClasses == i))
            N.Label = np.argmax(classDist)
            leftToExpand = leftToExpand[1:]
            continue
        
        # Test to see if nodes are leaves
        leftIsLeaf = False
        rightIsLeaf = False
        # Left child is a leaf, create leaf and don't enqueue
        for i in range(0,classCount):
            if((sum(abs(leftClasses-i)) == 0) and not leftIsLeaf):
                N.leftChild = Node(Label=i)
                leftIsLeaf = True
        # Left child isn't leaf, so enqueue node
        if(not leftIsLeaf):
            n = Node(X,Y,K,classCount,sortLeft)
            N.leftChild = n
            leftToExpand.append(n)
        # Right child is a leaf, create leaf and don't enqueue
        for i in range(0,classCount):
            if((sum(abs(rightClasses-i)) == 0) and not rightIsLeaf):
                N.rightChild = Node(Label=i)
                rightIsLeaf = True
        # Right child isn't leaf, so enqueue node
        if(not rightIsLeaf):
            n = Node(X,Y,K,classCount,sortRight)
            N.rightChild = n
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
        if(Label == None):
            # Claculate information gain - maximial feature and threshold
            self.currentSamples = currentSamples
            self.Thresh, self.feature = findBestF(X[currentSamples],Y[currentSamples],k,classCount)
    def test(self,X):
        # Sort all samples w.r.t. feature and threshold. Then take subset of example that are
        # actually at this node during constuction 
        sortLabels = X[:,self.feature] < self.Thresh
        right = np.logical_not(sortLabels)
        sortLeft = []
        sortRight = []
        for i in range(sortLabels.size):
            sortLeft.append(sortLabels[i].astype('int') * self.currentSamples[i].astype('int'))
            sortRight.append(right[i].astype('int') * self.currentSamples[i].astype('int'))

        sortLeft = np.array(sortLeft)
        sortRight = np.array(sortRight)
        return sortLeft.astype('bool'), sortRight.astype('bool')

    def sortTest(self,X):
        # Simply return thresholded array
        sortLabels = X[:,self.feature] < self.Thresh
        return sortLabels

def findBestF(X,Y,k,classCount):

    samples,parameters = np.shape(X)

    minH = float("infinity")
    bestF = []
    bestDiv = []

    # Choose a random k parameters to consider
    A = np.sort(X[:,range(k)],0)

    Thresholds = (A[:-1,:] + A[1:,:])/2

    for j in range(k):
        # Number of thresholds is 1/std for that column
        # If this feature has no variance, skip it
        if(STD[j] == 0):
            continue
        numOfThresh = int(1/max(STD[j],1.0/(samples-1)))
        s = np.linspace(0,samples-2,numOfThresh,dtype=int)
        for i in range(numOfThresh):
            # 1 if sample is less than threshold, 0 o.w.
            sortLabels = np.array(X[:,j] < Thresholds[s[i],j]).astype('int')

            HA = 0
            HB = 0 

            nA = sum(sortLabels)
            nB = samples - nA

            # Calculate entropy for current samples
            for y in range(0,classCount):
                classTruth = np.array(Y == y).astype('int')
                totalInClass = sum(classTruth)
                # 1 if in class y and sorted left, then summed
                pA = np.dot(classTruth,sortLabels)
                # All other must be in class y and sorted right
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
                bestT = Thresholds[s[i],j]
                bestF = j
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

