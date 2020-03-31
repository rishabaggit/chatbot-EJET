import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
import pickle

# Importing the dataset
dataset = pd.read_csv('train.csv')
train = dataset.iloc[:, 1:]
test = dataset.iloc[0:2001, 2:]

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

sentences = []
for i in range(0,len(train)):
    sent = re.sub(r"i'm", "i am", train['comment_text'][i])
    sent = re.sub(r"he's", "he is", train['comment_text'][i])
    sent = re.sub(r"she's", "she is", train['comment_text'][i])
    sent = re.sub(r"that's", "that is", train['comment_text'][i])
    sent = re.sub(r"what's", "what is", train['comment_text'][i])
    sent = re.sub(r"where's", "where is", train['comment_text'][i])
    sent = re.sub(r"how's", "how is", train['comment_text'][i])
    sent = re.sub(r"\'ll", " will", train['comment_text'][i])
    sent = re.sub(r"\'ve", " have", train['comment_text'][i])
    sent = re.sub(r"\'re", " are", train['comment_text'][i])
    sent = re.sub(r"\'d", " would", train['comment_text'][i])
    sent = re.sub(r"n't", " not", train['comment_text'][i])
    sent = re.sub(r"won't", "will not", train['comment_text'][i])
    sent = re.sub(r"can't", "cannot", train['comment_text'][i])
    sent = re.sub('[^A-Za-z0-9]+', ' ', train['comment_text'][i]) 
    sent = sent.lower()
    sent = sent.split()
    sent = [ps.stem(word) for word in sent if not word in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    sentences.append(sent)


'''for i in range(0, len(train)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)'''

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(sentences).toarray()
y = dataset.iloc[0:len(train), 2:].values

df_y = pd.DataFrame(columns=['class'])

for i in range(0,len(train)):
    if y[i][0] == 1 or y[i][1] == 1 or y[i][2] == 1 or y[i][3] == 1 or y[i][4] == 1 or y[i][5] or y[i][5] == 1:
        df_y.loc[i] = 1
    else:
        df_y.loc[i] = 0 
        
df_y=df_y.astype('int')


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size = 0.20, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

'''aved_classifier = 'saved_classifier.sav'
joblib.dump(classifier, saved_classifier)'''

joblib.dump(classifier, "class.pkl")
joblib.dump(cv, "my_cv.pkl")

from sklearn.metrics import confusion_matrix
y_train_prd = classifier.predict(X_train)
cm_train = confusion_matrix(y_train, y_train_prd)
acc_train = (float(cm_train[0][0]+cm_train[1][1])/(cm_train[0][0]+cm_train[1][1]+cm_train[0][1]+cm_train[1][0]))*100
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

acc = (float(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]))*100
precision = (float(cm[0][0])/(cm[0][0]+cm[0][1]))*100

print (acc, precision)
