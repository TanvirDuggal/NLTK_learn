# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:14:11 2020

@author: Tanvy
"""

import random
import nltk 
from nltk.corpus import movie_reviews


document = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(document)

allWords = []

for w in movie_reviews.words():
    allWords.append(w.lower())

allWords = nltk.FreqDist(allWords)

# print(allWords.keys())

wordFeatures = allWords.keys()
wordFeatures = list(wordFeatures)[:3000]

def find_features(doc):
    words    = set(doc)
    features = {}
    
    for w in wordFeatures:
        features[w] = (w in words)
        
    return features

featureSet = [(find_features(rev), category) for (rev, category) in document]
print(featureSet)