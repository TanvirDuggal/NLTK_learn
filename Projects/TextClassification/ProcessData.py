# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 00:52:58 2020

@author: Tanvy
"""

import pandas as pd
import numpy as np
import nltk
import re

from keras.preprocessing.text import Tokenizer

import spacy

nlp = spacy.load("en_core_web_sm")

class ProcessData:
    df     = ''
    txtSeq = []
    maxSeq = 0
    
    def __init__(self, df):
        self.df = df
    
    def connvertLowerCase(self):
        for i in range(len(self.df)):
            self.df.iloc[0]["review"] = self.df.iloc[0]["review"].lower()
    
    def removeUnwantedItems(self):
        for i in range(len(self.df)):
            d = self.df.iloc[i]["review"]
            d = d.replace("<br />", " ")
            d = re.sub('[^A-Za-z0-9]+', ' ', d)
            d = nltk.tokenize.word_tokenize(d)
            d = [word for word in d if word not in nlp.Defaults.stop_words] 
            self.df.iloc[i]["review"] = ' '.join(d)
    
    def lemmatization(self):
        for i in range(len(self.df)):
            d = self.df.iloc[i]["review"]
            txt = []
            for w in nlp(d):
                txt.append(w.lemma_)
            self.df.iloc[i]["review"] = ' '.join(txt)
    
    def tokenizing(self):
        tokenizer = Tokenizer()
        seq = []
        for i in range(len(self.df)):
            d = self.df.iloc[i]["review"]
            seq.append(d.split(" "))
        tokenizer.fit_on_texts(seq)
        self.txtSeq = tokenizer.texts_to_sequences(seq)
    
    def tfIdf(self):
        pass