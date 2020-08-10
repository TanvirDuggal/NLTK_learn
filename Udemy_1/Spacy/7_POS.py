# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:05:16 2020

@author: Tanvy
"""


import spacy

nlp = spacy.load("en_core_web_sm")

#---------- Baic example
doc1 = nlp("The lazy fox jumped over the dog.")
for token in doc1:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")
    
#---------- Compex with tenses
    
doc2 = nlp("I read a book.") # Paste tense
doc3 = nlp("I am reading a book.") # Present tense
doc4 = nlp("I read books.") #Paste tense on on 1 book

for token in doc2:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")
    
print("-----")

for token in doc3:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")

print("------")

for token in doc4:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}")
    
#------------- Counting POS and Tags
    
POS_count = doc1.count_by(spacy.attrs.POS)
print(POS_count)

for k, v in POS_count.items():
    print(f"{k} {doc1.vocab[k].text:{10}} {v:{10}}")
    
print("---------")

TAG_count = doc1.count_by(spacy.attrs.TAG)
print(TAG_count)

for k, v in TAG_count.items():
    print(f"{k:{10}} {doc1.vocab[k].text:{10}} {v:{10}}")
    
