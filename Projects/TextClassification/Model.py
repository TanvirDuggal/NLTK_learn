# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:36:50 2020

@author: Tanvy
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

import tensorflow as tf

class Model:
    classifier = ''
    
    def __init__(self):
        pass
    
    def createModel(self):
        classifier = Sequential()
        classifier.add(LSTM(units=50, return_sequences=True, input_shape=((1427, 1))))
        classifier.add(Dropout(0.3))
        classifier.add(LSTM(units=50, return_sequences=False))
        classifier.add(Dropout(0.3))
        classifier.add(Dense(50, activation='relu'))
        classifier.add(Dropout(0.3))
        classifier.add(Dropout(0.3))
        classifier.add(Dense(units=1))
        classifier.compile(optimizer='adam', loss='binary_crossentropy')
        self.classifier = classifier
        
    def fitModel(self, epochs, batchSize, X, Y):
        self.classifier.fit(X, Y, epochs=epochs, batch_size=batchSize)