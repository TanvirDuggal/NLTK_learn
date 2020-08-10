# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:11:02 2020

@author: Tanvy
"""


import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

doc = "Hello Everyone, my name is Tanvir Duggal"

# ------------- Tokenization text
for token in nlp(doc):
    print(token.text)
    
# ------------ Complex tokenization with punctuations
doc2 = "You can contact me on 437-772-4737, or email @ duggaltanvir@gmail.com"

for token in nlp(doc2):
    print(token.text)
    
# -------------- Named entity for text
doc3 = "Apple will invest $8 million in India"

for entity in nlp(doc3).ents:
    print(entity)
    print(entity.label_)
    print("")
    
# ------------- Visualizing text
    
doc4 = nlp("Apple if going to buy ABC for $9 million")

displacy.serve(doc4, style='dep')

displacy.serve(doc4, style="ent")