# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 00:07:03 2020

@author: Tanvy
"""


import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

def getEnt(doc):
    if doc.ents:
        for ent in doc.ents:
            print(f"{ent.text:{10}} {ent.label_:{10}} {spacy.explain(ent.label_)}")
    else:
        print("No Entity Found")
        
doc1 = nlp("Hello how are you")
getEnt(doc1)

doc2 = nlp("I am going to New Delhi, tomorrow and will visit Red Fort too. I will also meet James there")
getEnt(doc2)

# -------------------- Updating Named Entity list

doc2 = nlp("Tesla is great !!")
getEnt(doc2)

ORG    = doc2.vocab.strings[u"ORG"]
newEnt = Span(doc2, 0, 1, label=ORG)

doc2.ents = list(doc2.ents) + [newEnt]
getEnt(doc2)

# ----------------------- Updating multiple Entity List

doc3 = nlp(u"I am going to buy a new vacuum cleaner." 
           u"I bought a new vacuum-cleaner.")

getEnt(doc3)

matcher = PhraseMatcher(nlp.vocab)

phraseList = ['vacuum cleaner', 'vacuum-cleaner']
phrasePattern = [nlp(text) for text in phraseList]

matcher.add('newProduct', None, *phrasePattern)

foundMatcher = matcher(doc3)
print(foundMatcher)

PROD   = doc3.vocab.strings[u"PROD"]

for match in foundMatcher:
    print(match[1], match[2])

newEnt = [Span(doc3,match[1], match[2], label=PROD) for match in foundMatcher]

doc3.ents = list(doc3.ents) + newEnt

getEnt(doc3)


# ------------------ Displaying Named Entity

doc4 = nlp("Apple sold 20,000 iPods for $500 in the month of May")
displacy.serve(doc4, style="ent")

# ---------- Styling

options = {'ents':['ORG', 'PRODUCT'], 'color':{'ORG':'red', 'PRODUCT':'green'}}

displacy.serve(doc4, style="ent", options=options)