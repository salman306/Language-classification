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
tfidfTable=start.traincounts
TOTALTURN=len(tagArray)
ALPHABETS=start.headings
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

def getProbs_XjGivenYi():
    ALPHABETS_SIZE=len(ALPHABETS)
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
    weightCutoff=5
    for k,v in alphabetsFreqGivenYi.iteritems():
        for kk,vv in v.items():
            if (vv<weightCutoff):del alphabetsFreqGivenYi[k][kk]

    # re-weight with respect to the total alphabet of a language

    for k,v in alphabetsFreqGivenYi.iteritems():
        for kk,vv in v.items():
            #Laplace smoothing
            alphabetsFreqGivenYi[k][kk]=(1.0+vv)/(sum(v.values())+tagCount[k])
    return alphabetsFreqGivenYi
probXjGivenYi=getProbs_XjGivenYi()
def Predict(charArr):
    # make charArr contains unique chars since tf idf is already being considered
    charArr=start.np.unique(charArr)
    calculationDict=tagProb.copy()
    for k,v in calculationDict.items():
        tempSum=0
        for char in charArr:
            if (char in probXjGivenYi[k]):
                tempSum+=start.np.log(probXjGivenYi[k][char])
            else:
                tempSum+=start.np.log(1.0/(tagCount[k]+sum(probXjGivenYi[k].values())))
        calculationDict[k]=start.np.log(v)+tempSum
    # in prodcution mode,the line below should be commented out
    print(calculationDict)
    # return the language index where max log(probability) occurs
    return max(calculationDict,key=calculationDict.get)
