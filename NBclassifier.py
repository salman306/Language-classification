#!/usr/bin/env python
import start
from scipy.stats import itemfreq
import lr
import numpy as np
import pandas as pd
# reminder of the mapping
#mapping = {0: 'Slovak', 1: 'French', 2: 'Spanish', 3: 'German', 4: 'Polish'}
# the tags are stored in start.y_train
#i.e. tag for utterance index at 0 is start.y_train[0] or start.data['Target']
#def main():
TOTALTURN=len(start.data['Target'])
# trainPartitionEnd=int(np.floor(0.75*TOTALTURN))
tagArray=start.data['Target']
charArray=start.characterarray
# # trainArr_x=charArray[:trainPartitionEnd]
# crossArr_x=charArray[(trainPartitionEnd+1):]
# trainArr_y=tagArray[:trainPartitionEnd]
# crossArr_y=tagArray[(trainPartitionEnd+1):]

testSet=lr.testform
#function to get probability of Y_i's
# return a dictionary {0:0.333,1:0.2222}  for example
def getProbs_Yi(inputArr):
    #get the frequency of the tags
    typeCount=itemfreq(inputArr)
    #now typeCount looks like {0:111,1:222}
    #meaning there are 111 Slovak utterance and 222 French utterance
    typeCount=dict((x[0],x[1]) for x in typeCount)
    #P(Y_0)= typeProb[0]
    typeProb=dict((k,(v+0.0)/TOTALTURN) for k,v in typeCount.iteritems())
    return (typeCount,typeProb)
# tagCount,tagProb=getProbs_Yi(trainArr_y)
tagCount,tagProb=getProbs_Yi(tagArray)
languageCharCount=dict()
def getProbs_XjGivenYi():
    alphabetsFreqGivenYi=dict()
    for i in range(len(charArray)):
        #make the charList contain unique chars by turning them to set, note order is not preserved
        tmp=set(charArray[i])
        templength=len(tmp)
        #if the alphabetsFreqGivenYi[0:5] is not initiated with dictionary, then init with empty dict
        if(not alphabetsFreqGivenYi.has_key(tagArray[i])):
            alphabetsFreqGivenYi[tagArray[i]]=dict()
            languageCharCount[tagArray[i]]=templength
        else:
            languageCharCount[tagArray[i]]+=templength
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
        languageAlphabetLength=len(v)
        for kk,vv in v.iteritems():
            #Laplace smoothing
            alphabetsFreqGivenYi[k][kk]=(1.0+vv)/(languageCharCount[k]+languageAlphabetLength)
    return alphabetsFreqGivenYi
probXjGivenYi=getProbs_XjGivenYi()
# np.save("probCharBag.npy",probXjGivenYi)
np.save("charCount.npy",languageCharCount)
# probXjGivenYi=np.load('probCharBag.npy').item()
# probXjGivenYi=np.load('probs.npy').item()
"""
if we have difficulty distinguishing Fr or Sp
the following function is called
"""
def Sp_VS_Fr(characters):
    setCharacters=set(characters)
    SpScore=0
    FrScore=0
    for char in setCharacters:
        if (probXjGivenYi[1].has_key(char)):
            FrScore+=probXjGivenYi[1][char]
        # else:
        #     FrScore+=start.np.log(1.0/len(probXjGivenYi[1]))

        if (probXjGivenYi[2].has_key(char)):
            SpScore+=probXjGivenYi[2][char]
        # else:
        #     SpScore+=start.np.log(1.0/len(probXjGivenYi[2]))
    if(SpScore>FrScore):
        return 2
    return 1

def Predict(charArr):
    setCharArr=set(charArr)
    calculationDict=tagProb.copy()
    weight=1
    for k,v in calculationDict.iteritems():
        tempSum=0
        for char in setCharArr:
            if (probXjGivenYi[k].has_key(char)):
                tempSum+=start.np.log(probXjGivenYi[k][char])
            else:
                tempSum+=weight*start.np.log(1.0/(languageCharCount[k]+len(probXjGivenYi[k])))
        calculationDict[k]=start.np.log(v)+tempSum
    print(calculationDict)
    return max(calculationDict,key=calculationDict.get)
def Predict2(charArr):
    setCharArr=set(charArr)
    calculationDict=tagProb.copy()
    for k,v in calculationDict.iteritems():
        tempSum=0
        for char in setCharArr:
            if (probXjGivenYi[k].has_key(char)):
                tempSum+=start.np.log(probXjGivenYi[k][char])
            else:
                tempSum+=start.np.log(1.0/(languageCharCount[k]+len(probXjGivenYi[k])))
        calculationDict[k]=start.np.log(v)+tempSum
    res=max(calculationDict,key=calculationDict.get)
    if( (res==1 or res==2) and abs(calculationDict[1]-calculationDict[2])<0.8):
        return Sp_VS_Fr(charArr)
    return res
def predictWrap(charMatrix):
    result=[]
    for utt in charMatrix:
        result.append(Predict2(utt))
    return result
def diff(a,b):
    err=0.0
    for i in range(len(a)):
        if(not a[i]==b[i]):err+=1
    return err/len(a)
""" output the csv file"""
def CSVify(myList,name):
    df=pd.DataFrame(myList)
    df=df.reset_index()

    df.to_csv(name+'.csv',header=['Id','Category'],index=False)
t=predictWrap(testSet)
# diff(t,tagArray)
CSVify(t,"trainSet_CharBagwithDual")
