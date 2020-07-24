# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:15:29 2020

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset1=pd.read_csv('hateval2019_en_train.csv')
dataset2=pd.read_csv('hateval2019_en_test.csv')

#cleaning the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus1=[]
corpus2=[]
for i in range(0,9000):
    review=re.sub('[^a-zA-Z]', ' ',dataset1['text'][i])
    review=review.lower()
    review=review.split()
    ps= PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus1.append(review)

for i in range(0,3000):
    review=re.sub('[^a-zA-Z]', ' ',dataset2['text'][i])
    review=review.lower()
    review=review.split()
    ps= PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus2.append(review)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x_train=cv.fit_transform(corpus1).toarray()
x_test=cv.fit_transform(corpus2).toarray()
y_train=dataset1.iloc[:,2:3].values
y_test=dataset2.iloc[:,2:3].values

#creating tf-idf model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvectorizer1=TfidfVectorizer(max_features=1500)
x_train=tfidfvectorizer1.fit_transform(corpus1).toarray()
tfidfvectorizer2=TfidfVectorizer(max_features=1500)
x_test=tfidfvectorizer2.fit_transform(corpus2).toarray()
y_train=dataset1.iloc[:,2:3].values
y_test=dataset2.iloc[:,2:3].values

#creating word2vec model
from gensim import corpora,models,similarities
from gensim.models import KeyedVectors
v1=[nltk.word_tokenize(text) for text in corpus1]
v2=[nltk.word_tokenize(text) for text in corpus2]

from gensim.models import Word2Vec
model=Word2Vec(v1,min_count=2,window=5)
model.train(v1, total_examples=len(v1), epochs=40)
sentence1=[]
for text in corpus1:
    sen=np.zeros(100,dtype="float32")
    coun=0
    for word in text:
        try:
            sen+=model[word]
            coun+=1
        except KeyError:
            coun-=1
            continue;
    sentence1.append(sen)
x_train=sentence1

sentence2=[]
for text in corpus2:
    sen=np.zeros(100,dtype="float32")
    coun=0
    for word in text:
        try:
            sen+=model[word]
            coun+=1
        except KeyError:
            coun-=1
            continue;
    sentence2.append(sen)
x_test=sentence2

"""vectors=[]
for word in words:
         vectors.append(model[word])
x_train=np.array(vectors)

print(words)"""

y_train=dataset1.iloc[:,2:3].values
y_test=dataset2.iloc[:,2:3].values

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 3000, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

plt.hist(y_train)