# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 01:04:04 2020

@author: Tanvy
"""

from GetData import GetData

from ProcessData import ProcessData

from Model import Model

import pandas as pd
import numpy as np

import pickle

import sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import keras
from keras.preprocessing.sequence import pad_sequences

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

gd = GetData()
df = gd.movieDataFrame

for i in range(len(df)):
    dt = df.loc[i]["sentiment"]
    if dt == "positive":
        df.loc[i]["sentiment"] = "pos"
    elif dt == "negative":
        df.loc[i]["sentiment"] = "neg"

processor = ProcessData(df)

processor.connvertLowerCase()
processor.removeUnwantedItems()
processor.lemmatization()
processor.tokenizing()

seq = processor.txtSeq
processor.maxSeq = len(max(seq))
print(seq[0])

print(df["sentiment"].head())
print(pd.unique(df["sentiment"]))

labelEncoder = LabelEncoder()
df["sentiment"] = labelEncoder.fit_transform(df["sentiment"])

Y = df["sentiment"].values

print(df["sentiments"].head())
print(pd.unique(df["sentiment"]))

dt = [seq, Y]

print(Y)

with open("data.pickle", 'wb') as f:
    pickle.dump(dt, f)
    
d = ''

with open("data.pickle", 'rb') as f:
    d = pickle.load(f)
    
X = d[0]
Y = d[1]

maxLen = len(max(X,key = lambda x: len(x)))

x = []
y = []

for i in range(0, 100000, 10000):
    x.append(X[i:i+10000])
    y.append(Y[i:i+10000])    
 

def processData(X, Y):
    b = np.zeros([len(X),maxLen])
    for i,j in enumerate(X):
        b[i][0:len(j)] = j
    
    X = np.reshape(b, (b.shape[0], b.shape[1], 1))
    Y = np.reshape(Y, (b.shape[0], 1))
    return X, Y
    
X, Y = processData(x[0], y[0])

classifier = Model()
classifier.createModel()
classifier.fitModel(epochs=100, batchSize=32, X=X, Y=Y)
model = classifier.classifier

model.save("NLP_classifier.mdl")