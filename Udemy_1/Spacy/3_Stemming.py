# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:03:07 2020

@author: Tanvy
"""


words = ['run', 'runs', 'running', 'ran', 'kiss', 'kissed', 'kissing', 'fairness', 
         'heavly', 'slowly', 'shouting']

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

for word in words:
    print(word + " -> " + stemmer.stem(word))

print("----------")
    
from nltk.stem.snowball import SnowballStemmer
snowBall = SnowballStemmer(language='english')

for word in words:
    print(word + " -> " + snowBall.stem(word))