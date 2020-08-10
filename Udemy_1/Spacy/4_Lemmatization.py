# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:16:35 2020

@author: Tanvy
"""

import spacy

nlp = spacy.load("en_core_web_sm")

def show_lemma(docs):
    for doc in docs:
        print(f'{doc.text:{12}} {doc.pos_:{25}} {doc.lemma:{22}} {doc.lemma_:{24}} ')

doc1 = nlp("I took part in a race where I ran for 4 miles along with other runners")
show_lemma(doc1)