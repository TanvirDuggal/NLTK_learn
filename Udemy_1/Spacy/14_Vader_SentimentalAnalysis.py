# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 00:12:07 2020

@author: Tanvy
"""


import nltk
nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

sid = SentimentIntensityAnalyzer()

doc1 = "What an amazing movie it was, loved it"
print(sid.polarity_scores(doc1))

doc2 = "HATE this type of movie, I dont understand why people spends money on it !!!!!"
print(sid.polarity_scores(doc2))

df = pd.read_csv("amazonreviews.tsv", sep="\t")
print(df.head())

df["compoundScore"] = df["review"].apply(lambda review:sid.polarity_scores(review)["compound"])

print(df[["review", "compoundScore"]].head())

df["score"] = df["compoundScore"].apply(lambda score: 'pos' if score >= 0 else 'neg')

print(df.head())

print(confusion_matrix(df["label"], df["score"]))
print(accuracy_score(df["label"], df["score"]))