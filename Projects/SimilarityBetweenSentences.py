# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:24:28 2020

@author: Tanvy
"""


def getSimilarity(firstSentence, secondSentence):
    score     = 0
    
    splitSent = lambda s: set(s[i:i+3] for i in range(len(s)-2))
    jaccardDistance = lambda setA, setB : len(setA & setB)/float(len(setA | setB))

    try:
        score = jaccardDistance(splitSent(firstSentence), splitSent(secondSentence))
    except Exception as e:
        print(str(e))
        
    return score

firstSentence  = "welcome to medium"
secondSentence = "medium is a publishing platform"
print(getSimilarity(firstSentence, secondSentence))