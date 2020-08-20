# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 00:45:35 2020

@author: Tanvy
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("moviereviews.tsv", sep="\t")
print(df.head())

print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())

sid = SentimentIntensityAnalyzer()

df["compundScore"] = df["review"].apply(lambda text: sid.polarity_scores(text)["compound"])

print(df.head())

df["res"] = df["compundScore"].apply(lambda score: 'pos' if score >=0 else 'neg')

print(df.head())

print(confusion_matrix(df["label"], df["res"]))
print(accuracy_score(df["label"], df["res"]))