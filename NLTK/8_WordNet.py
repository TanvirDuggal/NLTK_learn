# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:45:29 2020

@author: Tanvy
"""

import nltk
from nltk.corpus import wordnet

def main():
    word = "good"
    syn  = set()
    ant  = set()
    # -------- antonyms synonyms 
    for s in wordnet.synsets(word):
        for l in s.lemmas():
            if l.name() != word:
                syn.add(l.name())
                
            if l.antonyms():
                ant.add(l.antonyms()[0].name())
          
    print(syn)
    print(ant)
    
    # ---------- Word Similarity
    
    w1 = "snake"
    w2 = "rat"
    
    w1_s = wordnet.synset(w1+".n.01")
    w2_s = wordnet.synset(w2+".n.01")
    
    print(w1_s.wup_similarity(w2_s))

if __name__ == '__main__':
    main()