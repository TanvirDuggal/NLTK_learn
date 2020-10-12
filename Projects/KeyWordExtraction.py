# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:54:20 2020

@author: Tanvy
"""


import spacy
from collections import Counter
from string import punctuation

nlp = spacy.load("en_core_web_lg")

def getHotWords(text):
    result  = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    
    doc     = nlp(text.lower())
    
    for token in doc:
        if token.text in punctuation or token.text in nlp.Defaults.stop_words:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)
            
    return result

text = ''' Welcome to Medium! Medium is a publishing platform where people can read important, insightful stories on the topics that matter most to them and share ideas with the world. '''

print(set(getHotWords(text)))