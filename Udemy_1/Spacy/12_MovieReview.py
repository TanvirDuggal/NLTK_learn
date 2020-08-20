# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:16:43 2020

@author: Tanvy
"""


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


df = pd.read_csv("moviereviews.tsv", sep="\t")

df.dropna(inplace=True)

X = df["review"]
Y = df["label"]

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.3, random_state=42)

pipe_clf = Pipeline([('tfidf'      , TfidfVectorizer()), 
                      ('classifier', LinearSVC())])


pipe_clf.fit(X_train, Y_train)

predict = pipe_clf.predict(X_test)

print(confusion_matrix(Y_test, predict))

print(classification_report(Y_test, predict))

print(accuracy_score(Y_test, predict))
