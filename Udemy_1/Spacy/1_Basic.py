# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:20:29 2020

@author: Tanvy
"""


import spacy
from spacy import displacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

text = "Hello world this is the first Spacy program I am writing"
doc  = nlp(text)

print(doc)

for token in doc:
    print(token.text, token.pos, token.pos_)
    
doc2 = nlp("A great man once said 'Can you borrow?'")

print(type(doc2))
doc2_s = doc2[22:36]
print(doc2_s)
print(type(doc2_s))

doc3 = nlp("This is first sentence. This is second. This is third")

for token in doc3.sents:
    print(token)
    