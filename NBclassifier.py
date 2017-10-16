#!/usr/bin/env python
import start
from scipy.stats import itemfreq
# reminder of the mapping
#mapping = {0: 'Slovak', 1: 'French', 2: 'Spanish', 3: 'German', 4: 'Polish'}
# the tags are stored in start.y_train
#i.e. tag for utterance index at 0 is start.y_train[0] or start.data['Target']
#def main():
tagArray=start.data['Target']
charArray=start.characterarray
TOTALTURN=len(tagArray)
#function to get probability of Y_i's
# return a dictionary {0:0.333,1:0.2222}  for example
def getProbs_Yi():
    #get the frequency of the tags
    typeCount=itemfreq(tagArray)
    #now typeCount looks like {0:111,1:222}
    #meaning there are 111 Slovak utterance and 222 French utterance
    typeCount=dict((x[0],x[1]) for x in typeCount)
    #P(Y_0)= typeProb[0]
    typeProb=dict((k,(v+0.0)/TOTALTURN) for k,v in typeCount.iteritems())
    return (typeCount,typeProb)
tagCount,tagProb=getProbs_Yi()
def getProbs_XjGivenYi():
    alphabetsFreqGivenYi=dict()
    for i in range(TOTALTURN):
        #make the charList contain unique chars by turning them to set, note order is not preserved
        tmp=set(charArray[i])
        #if the alphabetsFreqGivenYi[0:5] is not initiated with dictionary, then init with empty dict
        if(not alphabetsFreqGivenYi.has_key(tagArray[i])):
            alphabetsFreqGivenYi[tagArray[i]]=dict()
        # for every char in an utterance count 1 appearance for the given language
        for char in tmp:
            if (alphabetsFreqGivenYi[tagArray[i]].has_key(char)):
                alphabetsFreqGivenYi[tagArray[i]][char]+=1
            else:
                alphabetsFreqGivenYi[tagArray[i]][char]=1
                """
                the data will look like at this step:
                alphabetsFreqGivenYi[0]={'a':200,'d':111}
                alphabetsFreqGivenYi[1]={'e':200,'q':111}
                """
    # change them to probability
    for k,v in alphabetsFreqGivenYi.iteritems():
        for kk,vv in v.iteritems():
            #Laplace smoothing
            alphabetsFreqGivenYi[k][kk]=(1.0+vv)/(tagCount[k]+2)
    return alphabetsFreqGivenYi
probXjGivenYi=getProbs_XjGivenYi()
def Predict(charArr):
    setCharArr=set(charArr)
    calculationDict=tagProb.copy()
    for k,v in calculationDict.iteritems():
        tempSum=0
        for char in setCharArr:
            if (probXjGivenYi[k].has_key(char)):
                tempSum+=start.np.log(probXjGivenYi[k][char])
            else:
                tempSum+=start.np.log(1.0/(tagCount[k]+2))
        calculationDict[k]=start.np.log(v)+tempSum
    print(calculationDict)
    return max(calculationDict,key=calculationDict.get)
