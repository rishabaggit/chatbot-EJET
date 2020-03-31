#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import re

from sklearn.externals import joblib
import pickle
#saved_classifier = joblib.load('saved_classifier.sav')

saved_classifier = joblib.load("class.pkl")
my_cv = joblib.load("my_cv.pkl")

'''import re
import nltk
nltk.download('stopwords')'''
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#dataset = pd.read_csv('train.csv')
#train = dataset.iloc[0:20001, 1:]
#y = dataset.iloc[0:len(train), 2:].values

sentences = []
'''for i in range(0,len(train)):
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
    sentences.append(sent)'''
    

'''for i in range(0, len(train)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)'''

'''from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(sentences).toarray()
y = dataset.iloc[0:len(train), 2:].values'''

#df_y = pd.DataFrame(columns=['class'])
#
#for i in range(0,len(train)):
#    if y[i][0] == 1 or y[i][1] == 1 or y[i][2] == 1 or y[i][3] == 1 or y[i][4] == 1 or y[i][5] or y[i][5] == 1:
#        df_y.loc[i] = 1
#    else:
#        df_y.loc[i] = 0 
#        
#df_y=df_y.astype('int')

input_sentence = input('please enter the comment : ')

mysentences = []

sent = re.sub(r"i'm", "i am", input_sentence)
sent = re.sub(r"he's", "he is", input_sentence)
sent = re.sub(r"she's", "she is", input_sentence)
sent = re.sub(r"that's", "that is", input_sentence)
sent = re.sub(r"what's", "what is", input_sentence)
sent = re.sub(r"where's", "where is", input_sentence)
sent = re.sub(r"how's", "how is", input_sentence)
sent = re.sub(r"\'ll", " will", input_sentence)
sent = re.sub(r"\'ve", " have", input_sentence)
sent = re.sub(r"\'re", " are", input_sentence)
sent = re.sub(r"\'d", " would", input_sentence)
sent = re.sub(r"n't", " not", input_sentence)
sent = re.sub(r"won't", "will not", input_sentence)
sent = re.sub(r"can't", "cannot", input_sentence)
sent = re.sub('[^A-Za-z0-9]+', ' ', input_sentence) 
sent = sent.lower()
sent = sent.split()
sent = [ps.stem(word) for word in sent if not word in set(stopwords.words('english'))]
sent = ' '.join(sent)
mysentences.append(sent)

X_my = my_cv.transform(mysentences).toarray()
y_pred_my = saved_classifier.predict(X_my)
print (int(y_pred_my))