#!/usr/bin/env python
import start
from scipy.stats import itemfreq
import lr
import pandas as pd
import numpy as np
"""
note I changed something on the lr.py I use

"""

# reminder of the mapping
#mapping = {0: 'Slovak', 1: 'French', 2: 'Spanish', 3: 'German', 4: 'Polish'}
# the tags are stored in start.y_train
#i.e. tag for utterance index at 0 is start.y_train[0] or start.data['Target']
#def main():
tagArray=start.data['Target']
charArray=start.characterarray
tfidfTable=start.traincounts
#tfidfTable4Test=lr.testcounts
TOTALTURN=len(tagArray)
ALPHABETS=start.headings
testChar=lr.testform
#function to get probability of Y_i's
# return a dictionary {0:0.333,1:0.2222}  for example
def getProbs_Yi():
    #get the frequency of the tags
    typeCount=itemfreq(tagArray)
    #now typeCount looks like {0:111,1:222}
    #meaning there are 111 Slovak utterance and 222 French utterance
    typeCount=dict((x[0],x[1]) for x in typeCount)
    #P(Y_0)= typeProb[0]
    typeProb=dict((k,(v+0.0)/TOTALTURN) for k,v in typeCount.items())
    return (typeCount,typeProb)
tagCount,tagProb=getProbs_Yi()
""" with Laplace smoothing
p(char='a'|language=1)=
(sum(tfidf_Weights of 'a' for all language=1 )+1)
/(
total size of utterance for language=1 +sum(tfidf_Weights of 'a' for all language=1 )
+sum(tfidf_Weights of 'b' for all language=1 )+... (for all alphabet of language=1)
)
"""
"""
Note currently I cached the probability table in a file so that it is faster to run the program
If you want to redesign the algo for prob table then u need to comment out line 90 and
uncomment line 88 after u did something in getProbs_XjGivenYi funciton
"""
languageSum=dict()
def getProbs_XjGivenYi():
    alphabetsFreqGivenYi=dict()
    for i in range(TOTALTURN):
        #cast the utterance charArray to set
        tmp=start.np.unique(charArray[i])
        #if the alphabetsFreqGivenYi[0:5] is not initiated with dictionary, then init with empty dict
        if(tagArray[i] not in alphabetsFreqGivenYi):
            alphabetsFreqGivenYi[tagArray[i]]=dict()
        for char in tmp:
            #change the encoding is critical)
            if (char in alphabetsFreqGivenYi[tagArray[i]]):
                alphabetsFreqGivenYi[tagArray[i]][char]+=tfidfTable[i].toarray()[0][ALPHABETS[char]]
            else:
                if(char not in ALPHABETS):
                    print("missed")
                    print((str(i)))
                    continue
                alphabetsFreqGivenYi[tagArray[i]][char]=tfidfTable[i].toarray()[0][ALPHABETS[char]]
                """
                the data will look like at this step: numbers below are just meant as example, not real data
                alphabetsFreqGivenYi[0]={'a':11.35,'d':22.2} in this case sum(tfidf_Weights of 'a' for all language=0 )=11.35
                alphabetsFreqGivenYi[1]={'e':13.21,'q':.5}
                """
    #get rid off weight that is too small, they should be treated as noise
    #after careful examination 5 is chosen
    # weightCutoff=5
    # for k,v in alphabetsFreqGivenYi.iteritems():
    #     for kk,vv in v.items():
    #         if (vv<weightCutoff):del alphabetsFreqGivenYi[k][kk]

    # re-weight with respect to the total alphabet of a language

    for k,v in alphabetsFreqGivenYi.iteritems():
        tempVSum=sum(v.values())
        languageSum[k]=tempVSum
        for kk,vv in v.items():
            #Laplace smoothing
            alphabetsFreqGivenYi[k][kk]=(1.0+vv)/(tempVSum+tagCount[k])
    #rescale so that prob (Xj=1:m|Yi)sums to 1
    for k,v in alphabetsFreqGivenYi.iteritems():
        tempVSum=sum(v.values())
        for kk,vv in v.items():
            #Laplace smoothing
            alphabetsFreqGivenYi[k][kk]=vv/tempVSum
    return alphabetsFreqGivenYi
# probXjGivenYi=getProbs_XjGivenYi()
# np.save("probtfidf.npy",probXjGivenYi)
# np.save("weightSumtfidf.npy",languageSum)
#probability is cached in file
probXjGivenYi=np.load('probtfidf.npy').item()
languageSum=np.load("weightSumtfidf.npy").item()
#index is the index of the charArr of the whole testset, tfidfTest is the tfidfTable of the testset
"""the functions below are using tfidf re-weight on test set
    result not good, need investigation
"""
def Predict(charArr,index,tfidfTest):
    # make charArr contains unique chars since tf idf is already being considered
    charArr=start.np.unique(charArr)
    calculationDict=tagProb.copy()
    for k,v in calculationDict.items():
        tempSum=0
        for char in charArr:
            if (char in probXjGivenYi[k]):
                #weight the charArray to be predicted by the the test tfidf weight
                weight=tfidfTest[index].toarray()[0][ALPHABETS[char]]
                #deals with the situation when product is numerically 0 and thus log will fail
                product=weight*probXjGivenYi[k][char]
                if(product==0):
                    product=1.0/(tagCount[k]+languageSum[k])
                tempSum+=start.np.log(product)
            else:
                # saves time by not doing summation since the summation will be 1
                # tempSum+=start.np.log(1/(tagCount[k]+sum(probXjGivenYi[k].values())))
                tempSum+=start.np.log(1.0/(tagCount[k]+languageSum[k]))
        calculationDict[k]=start.np.log(v)+tempSum
    # in prodcution mode,the line below should be commented out
    # print(calculationDict)
    # return the language index where max log(probability) occurs
    return max(calculationDict,key=calculationDict.get)
def predictWrap(charMatrix,tTable):
    result=[]
    i=0
    for utt in charMatrix:
        result.append(Predict(utt,i,tTable))
        i+=1
    return result
# t=predictWrap(testChar,tfidfTable4Test)
def Sp_VS_Fr(characters):
    setCharacters=set(characters)
    SpScore=0
    FrScore=0
    for char in setCharacters:
        if (probXjGivenYi[1].has_key(char)):
            FrScore+=start.np.log(probXjGivenYi[1][char])
        else:
            FrScore+=start.np.log(1.0/(tagCount[1]+languageSum[1]))
        if (probXjGivenYi[2].has_key(char)):
            SpScore+=start.np.log(probXjGivenYi[2][char])
        else:
            SpScore+=start.np.log(1.0/(tagCount[2]+languageSum[2]))
    if(SpScore>FrScore):
        return 2
    return 1


"""these functions below does not do use tfidf weight for prediction
and are currently used
"""
def Predict2(charArr):
    # make charArr contains unique chars since tf idf is already being considered
    charArr=start.np.unique(charArr)
    calculationDict=tagProb.copy()
    for k,v in calculationDict.items():
        tempSum=0
        for char in charArr:
            if (char in probXjGivenYi[k]):
                product=probXjGivenYi[k][char]
                if(product==0):
                    product=1.0/(tagCount[k]+languageSum[k])
                tempSum+=start.np.log(product)
            else:
                # saves time by not doing summation since the summation will be 1
                # tempSum+=start.np.log(1/(tagCount[k]+sum(probXjGivenYi[k].values())))
                tempSum+=start.np.log(1.0/(tagCount[k]+languageSum[k]))
        calculationDict[k]=start.np.log(v)+tempSum
    # in prodcution mode,the line below should be commented out
    # print(calculationDict)
    # return the language index where max log(probability) occurs
    res= max(calculationDict,key=calculationDict.get)
    if( (res==1 or res==2) and abs(calculationDict[1]-calculationDict[2])<0.0):
        return Sp_VS_Fr(charArr)
    return res
def predictWrap2(charMatrix):
    result=[]
    for utt in charMatrix:
        result.append(Predict2(utt))
    return result

"""
run prediction on testSet
data is in a list called testSetResult
"""
testSetResult2=predictWrap2(testChar)


"""
 a function to test results
a,b are lists of same size, report the err rate
this function is used to test error rate between two prediction
"""
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


"""Compare tfidf NB vs tfidf LogisticRegression"""
dfLogi=pd.DataFrame.from_csv('submission.csv')
#read the prediction data of LogisticRegression in dfLogi
dfLogi=dfLogi['0']
""" this is the difference between NB and Logistic Regression """
print(diff(testSetResult2,dfLogi))
CSVify(testSetResult2,'NoReweightWithDual0pcent')
