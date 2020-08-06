# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:55:09 2020

@author: Tanvy
"""


import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

def main():
    train_text  = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")
    
    # print(train_text)
    
    sentTokenizer = PunktSentenceTokenizer(train_text)
    tokenize      = sentTokenizer.tokenize(sample_text)
    
    for i in tokenize:
        # print(i)
        words = nltk.word_tokenize(i)
        pos   = nltk.pos_tag(words)
        print(">>> ", pos)

if __name__ == '__main__':
    main()