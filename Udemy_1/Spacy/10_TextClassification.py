# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:13:06 2020

@author: Tanvy
"""


import numpy as np
import pandas as pd

df = pd.read_csv("smsspamcollection.tsv", sep='\t')
print(df.head())
print(df.isnull().sum())

import sklearn
from sklearn.model_selection import train_test_split

X = df[["length", "punct"]]
Y = df["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(X_train.head())

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()
regressor.fit(X_train, Y_train)

predict = regressor.predict(X_test)

from sklearn import metrics

cf = metrics.confusion_matrix(predict, Y_test)
print(cf)

acc = metrics.accuracy_score(Y_test, predict)
print(acc)