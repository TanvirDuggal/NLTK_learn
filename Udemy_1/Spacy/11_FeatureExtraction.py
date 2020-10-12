# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:14:44 2020

@author: Tanvy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("smsspamcollection.tsv", sep='\t')
print(df.head())
print(df.isnull().sum())

X = df["message"]
Y = df["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

print(X_train.head())
print(X_train[0])
print(X_train_count[0])

tfidf_transformer = TfidfTransformer()
X_train_Tfidf     = tfidf_transformer.fit_transform(X_train_count)

print(X_train_Tfidf.shape)

vectorizer = TfidfVectorizer()
X_Tfidf    = vectorizer.fit_transform(X_train)

print(X_Tfidf.shape)

classifier = LinearSVC()
classifier.fit(X_Tfidf, Y_train)

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit(X_train, Y_train)

predict = text_clf.predict(X_test)

print(confusion_matrix(Y_test, predict))

print(classification_report(Y_test, predict))

print(accuracy_score(Y_test, predict))

print(text_clf.predict(["Hey how are you"]))

print(text_clf.predict(["Hello you just won $20,000, please conform with you phone number and email address"]))