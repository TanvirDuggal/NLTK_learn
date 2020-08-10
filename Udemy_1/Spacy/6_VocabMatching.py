# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:37:44 2020

@author: Tanvy
"""


import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)

pattern1 = [{'LOWER':'solarpower'}]
pattern2 = [{'LOWER':'solar'}, {'IS_PUNCT':True}, {'LOWER':'power'}]
pattern3 = [{'LOWER':'solar'}, {'LOWER':'power'}]

matcher.add('Solarpower', None, pattern1, pattern2, pattern3)

doc1 = nlp("Solar power is future, the power of solar-power can lead to many great lives")

found_match = matcher(doc1)

print(found_match)