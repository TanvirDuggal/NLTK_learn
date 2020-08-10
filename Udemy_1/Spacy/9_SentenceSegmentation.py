# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:01:11 2020

@author: Tanvy
"""


import spacy
nlp = spacy.load("en_core_web_sm")

doc1 = nlp("This is first sentence. This is second sentence. This is third sentence")

for line in doc1.sents:
    print(line)
    
# ------------ For adding new rule
    
def newSeg(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc

doc2 = nlp("This is first sentense; This is second sentence. This is third sentence")
for line in doc2.sents:
    print(line)

nlp.add_pipe(newSeg, before="parser")
print(nlp.pipe_names)

for line in doc2.sents:
    print(line)