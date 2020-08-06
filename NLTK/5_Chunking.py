# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:31:30 2020

@author: Tanvy
"""

import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union

def main():
    train_text  = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")
    
    # print(train_text)
    
    sentTokenizer = PunktSentenceTokenizer(train_text)
    tokenize      = sentTokenizer.tokenize(sample_text)
    
    for i in tokenize:
        words   = nltk.word_tokenize(i)
        pos     = nltk.pos_tag(words)
        chunkG  = """ Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?} """
        chunked = nltk.RegexpParser(chunkG)
        chunk   = chunked.parse(chunked)
        
        chunk.draw()

if __name__ == '__main__':
    main()
