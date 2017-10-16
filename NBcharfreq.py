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
languageWordCount=dict()
# ALPHABETS=start.headings
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
    # count how many words of language 0:4
    typeWordCount=tagCount.copy()
    for k,v in typeWordCount.iteritems():
        typeWordCount[k]=0
    for i in range(TOTALTURN):
        tmp=charArray[i]
        tmplen=len(charArray)
        typeWordCount[tagArray[i]]+=tmplen
        #if the alphabetsFreqGivenYi[0:5] is not initiated with dictionary, then init with empty dict
        if(not alphabetsFreqGivenYi.has_key(tagArray[i])):
            alphabetsFreqGivenYi[tagArray[i]]=dict()
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
            alphabetsFreqGivenYi[k][kk]=(1.0+vv)/(typeWordCount[k]+2)
    return alphabetsFreqGivenYi,typeWordCount
(probXjGivenYi,languageWordCount)=getProbs_XjGivenYi()
def Predict(charArr):
    # setCharArr=set(charArr)
    calculationDict=tagProb.copy()
    for k,v in calculationDict.iteritems():
        tempSum=0
        for char in charArr:
            if (probXjGivenYi[k].has_key(char)):
                tempSum+=start.np.log(probXjGivenYi[k][char])
            else:
                tempSum+=start.np.log(1.0/(languageWordCount[k]+2))
        calculationDict[k]=start.np.log(v)+tempSum
    print(calculationDict)
    return max(calculationDict,key=calculationDict.get)
