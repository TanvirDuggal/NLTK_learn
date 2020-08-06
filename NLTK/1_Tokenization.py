# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:40:15 2020

@author: Tanvy
"""


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

text = "The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem.[2] However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed."
# --------- Tokenization
print(sent_tokenize(text))
print(word_tokenize(text))

text = "Hello everyone, my name is Tanvir Duggal. I am 26 years old. I work at KPMG"
print(nltk.tokenize.MWETokenizer(word_tokenize(text)).tokenize(text))
