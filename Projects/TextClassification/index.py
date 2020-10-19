# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 01:04:04 2020

@author: Tanvy
"""

from GetData import GetData
from ProcessData import ProcessData
import pandas as pd
import numpy as np

gd = GetData()
df = gd.movieDataFrame

for i in range(len(df)):
    dt = df.loc[i]["sentiment"]
    if dt == "positive":
        df.loc[i]["sentiment"] = "pos"
    elif dt == "negative":
        df.loc[i]["sentiment"] = "neg"

processor = ProcessData(df)

processor.connvertLowerCase()
processor.removeUnwantedItems()
processor.lemmatization()
processor.tokenizing()

seq = processor.txtSeq

print(seq[0])

