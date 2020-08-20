# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:50:28 2020

@author: Tanvy
"""


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

npr = pd.read_csv("npr.csv")
tfidf = TfidfVectorizer(max_df=0.95, min_df = 2, stop_words="english")

dtm = tfidf.fit_transform(npr["Article"])

nmf_model = NMF(n_components=7, random_state=42)
nmf_model.fit(dtm)

for i, topic in enumerate(nmf_model.components_):
    print("TOP 10 WORDS FOR TOPIC " + str(i))
    print([tfidf.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print("\n")
    
topic_result = nmf_model.transform(dtm)
topic_result.argmax(axis=1)

npr["Topic"] = topic_result.argmax(axis=1)
npr.head()