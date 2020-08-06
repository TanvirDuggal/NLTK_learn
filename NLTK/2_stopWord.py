# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:15:11 2020

@author: Tanvy
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed."
# print(text)

stopWords = stopwords.words("english")
text      = word_tokenize(text)

# print(stopWords)
print(text)

filteredText = [x for x in text if x not in stopWords]

print(filteredText)