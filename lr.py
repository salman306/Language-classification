
seed = 1
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets, metrics
import matplotlib.pyplot as plt

def cleaner(data):
    data['Text'] = map(str, data['Text'])
    temp = []
    characterarrayfunc = []
    for utterance in data['Text']:
        for counter in utterance:
            if ((counter != ' ') and (counter.isdigit() == False) and (counter != '.')):
                if counter.isupper():
                    temp.append(counter.lower())
                else:
                    temp.append(counter)
        characterarrayfunc.append(temp)
        temp = []

    finalform = []
    for counter2 in characterarrayfunc:
        finalform.append(''.join(counter2))

    return finalform



data = pd.DataFrame.from_csv('train_set_x.csv')
trainresults = pd.DataFrame.from_csv('train_set_y.csv')
testx = pd.DataFrame.from_csv('test_set_x.csv')
testy = pd.DataFrame.from_csv('random_baseline.csv')


data['Target'] = trainresults['Category']
mapping = {0: 'Slovak', 1: 'French', 2: 'Spanish', 3: 'German', 4: 'Polish'}

# removing blanks


# converting all to Text


# removing spaces between words to create list of characters
finalform = cleaner(data)
testform = cleaner(testx)

count_vect = TfidfVectorizer(decode_error= 'ignore', analyzer = 'char', use_idf=True)
traincounts = count_vect.fit_transform(finalform)

#feature headings i.e. the alphabets in the corpus.
headings = count_vect.vocabulary_


#data is divided into training and validation sets
#I will reformat the test set when we sit down to test the algos together so we can benchmark the final algos

X_train, X_valid, y_train, y_valid = train_test_split(traincounts, np.array(data['Target']), test_size=0.00, random_state = seed)

logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)

testcounts = count_vect.transform(testform)
test_y = np.array(testy['Category'])


print('Training accuracy', metrics.accuracy_score(y_train, logreg.predict(X_train)))
#print('Validation accuracy', metrics.accuracy_score(y_valid, logreg.predict(X_valid)))
#print('Test accuracy', metrics.accuracy_score(test_y, logreg.predict(testcounts)))

df = pd.DataFrame(logreg.predict(testcounts))
df.rename = ['Id', 'Category']
df.to_csv('submission.csv')
