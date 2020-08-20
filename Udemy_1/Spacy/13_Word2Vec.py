# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:40:28 2020

@author: Tanvy
"""


import spacy
nlp = spacy.load('en_core_web_lg')

from scipy import spatial

# --------- Getting vector form of words and doc
vec = nlp('king').vector
print(vec.shape)

doc = nlp('The fox jumped over the lazy lion')
print(doc.vector)
print(doc.vector.shape)


# ----------- Getting relationship between words

st = nlp('king queen soldier')

for token in st:
    for token2 in st:
        print(token , " " , token2 , " " , token.similarity(token2))
        
#------- All the vocab
        
print(len(nlp.vocab.vectors))

# ---------- Check if vec is in vocab

doc = nlp("king queen soldier nargil")

for token in doc:
    print(token, " ", token.has_vector, " ", token.vector_norm, " ", token.is_oov)
    
# ----- Similarity between words
    
king  = nlp.vocab['king'].vector
man   = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

cosin_simi = lambda vec1, vec2 :1-spatial.distance.cosine(vec1, vec2)

newVector = king-man+woman

simil = []

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                s_word = cosin_simi(newVector, word.vector)
                simil.append((word, s_word))
                
computed_simil = sorted(simil, key=lambda item:-item[1])
print([t[0].text for t in computed_simil[:10]])