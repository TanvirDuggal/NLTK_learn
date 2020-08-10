# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:28:28 2020

@author: Tanvy
"""


import spacy

nlp = spacy.load("en_core_web_sm")

print(nlp.Defaults.stop_words)
print(len(nlp.Defaults.stop_words))

print(nlp.vocab["is"].is_stop)