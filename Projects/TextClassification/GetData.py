# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:15:13 2020

@author: Tanvy
"""


import pandas as pd
import numpy as np
import os

class GetData:
    movieDataFrame = ''
    
    def __init__(self):
        self.getData()
        
    def getData(self):
        df1 = pd.read_csv("archive/IMDB Dataset.csv")
        imbdList       = []
        try:
            path = 'aclImdb_v1/aclImdb'
            for tt in ['test', 'train']:
                for negPos in ['neg', 'pos']:
                    fullPath = path + '/' + tt + '/' + negPos
                    for f in os.listdir(fullPath):
                        with open(fullPath+'/' + f, 'r', encoding="utf8") as ff:
                            imbdList.append([ff.readline(), negPos])
        except Exception as e:
            print(str(e))
                
        
        df2 = pd.DataFrame(imbdList, columns=['review','sentiment'])
        df = df1.append(df2, ignore_index=True)
         
        self.movieDataFrame = df
        
        
dg = GetData()