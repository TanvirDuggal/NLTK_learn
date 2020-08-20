# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:51:55 2020

@author: Tanvy
"""


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

npr = pd.read_csv("npr.csv")
print(npr.head())

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
dtm = vectorizer.fit_transform(npr["Article"])

print(dtm.shape)

LDA = LatentDirichletAllocation(n_components=7, random_state=42)

LDA.fit(dtm)

print(len(LDA.components_))

print(LDA.components_[0])
print(LDA.components_.shape)

single_topic  = LDA.components_[0]
single_topic.argsort()

print(single_topic)

top_words = single_topic.argsort()[-20:]

for word in top_words:
    print(vectorizer.get_feature_names()[word])
    
for i, topic in enumerate(LDA.components_):
    print("TOP 15 Words for Topic : " + str(i))
    print([vectorizer.get_feature_names()[index] for index in topic.argsort()[-10:]])
    print("\n")
    
topic_result = LDA.transform(dtm)

print(topic_result[0].round(2))

npr["topic"] = topic_result.argmax(axis=1)

print(npr.head())