# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:11:31 2020

@author: Tanvy
"""

import nltk
from nltk.stem import WordNetLemmatizer

def main():
    lem = WordNetLemmatizer()
    print(lem.lemmatize("goods"))
    
    print(lem.lemmatize("cacti"))
    print(lem.lemmatize("runs"))
    print(lem.lemmatize("grabbed", pos='v'))

if __name__ == '__main__':
    main()