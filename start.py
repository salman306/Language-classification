
seed = 1
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split


data = pd.DataFrame.from_csv('train_set_x.csv')
trainresults = pd.DataFrame.from_csv('train_set_y.csv')
testx = pd.DataFrame.from_csv('test_set_x.csv')

data['Target'] = trainresults['Category']
mapping = {0: 'Slovak', 1: 'French', 2: 'Spanish', 3: 'German', 4: 'Polish'}

# removing blanks
data = data[data['Text'] != '']


# converting all to Text
data['Text'] = map(str, data['Text'])


# removing spaces between words to create list of characters
temp = []
characterarray = []
alphabetSet=set()
for utterance in data['Text']:
    for counter in utterance:
        if ((counter != ' ') and (counter.isdigit() == False) and (counter != '.')):
            if counter.isupper():
                temp.append(counter.lower())
                alphabetSet.add(counter.lower())
            else:
                temp.append(counter)
                alphabetSet.add(counter.lower())
    characterarray.append(temp)
    temp = []
alphabetSet=list(alphabetSet)
alphabetSet.sort()
alphabetMapping=dict()
for i in range(len(alphabetSet)):
    alphabetMapping[alphabetSet[i]]=i 

finalform = []
for counter in characterarray:
    finalform.append(''.join(counter))

count_vect = TfidfVectorizer(decode_error= 'ignore', analyzer = 'char')
traincounts = count_vect.fit_transform(finalform)

#feature headings i.e. the alphabets in the corpus.
headings = count_vect.vocabulary_


#data is divided into training and validation sets
#I will reformat the test set when we sit down to test the algos together so we can benchmark the final algos

X_train, X_valid, y_train, y_valid = train_test_split(traincounts, np.array(data['Target']), test_size=0.20, random_state = seed)
