# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:33:20 2020

@author: Tanvy
"""

import spacy
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

from keras.preprocessing.sequence import pad_sequences

import numpy as np

from pickle import load, dump

def getFiles(filePath):
    with open(filePath) as f:
        str_u = f.read()
    return str_u

nlp = spacy.load('en', disable=['parser','tagger','ner'])
nlp.max_length = 1198623

def seperatePunc(text):
    return [token.text.lower() for token in nlp(text) if token.text not in '\n\n \n\n\n!\"-#$%&()--.*+,-/:;<=>?@[\\\\]^_`{|}~\\t\\n']
            
d = getFiles('moby_dick_four_chapters.txt')
token = seperatePunc(d)

print(len(token))

textSeq = []

textLen = 25+1

for i in range(textLen, len(token)):
    seq = token[i-textLen:i]
    textSeq.append(seq)
    
print(textSeq[0])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(textSeq)

seq = tokenizer.texts_to_sequences(textSeq)

print(seq[0])

print(tokenizer.index_word)

print(seq[0])

seq = np.array(seq)

print(seq[0])

vocabSize = len(tokenizer.word_counts)
print(vocabSize)

X = seq[:, :-1]
Y = seq[:, -1]

Y = to_categorical(Y, num_classes = vocabSize+1)
seqLen = X[1].shape
print(Y[0])
print(seqLen)

def createModel(vocabSize, seqLen):
    model = Sequential()
    model.add(Embedding(vocabSize, seqLen[0], input_shape=seqLen))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(vocabSize, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

model = createModel(vocabSize+1, seqLen)

model.fit(X, Y, batch_size=32, epochs=2, verbose=1)

model.save("model_mobyDick.mdl")

dump(tokenizer, open("myTokenizer", 'wb'))

def generateText(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text  = seed_text
    
    for i in range(num_gen_words):
#         -------- Tokenizing i.e converting words to num
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
#         ----------- If text is long or short it encoded them
        pad_encode   = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
#         ------------- Predicting class i.e the index of word and getting them
        pre_word_ind = model.predict_classes(pad_encode, verbose=0)[0]
        pred_word    = tokenizer.index_word[pre_word_ind]
        input_text  += ' ' + pred_word
        output_text.append(pred_word)

    return  ' '.join(output_text)


seed_text = textSeq[2]
seed_text = ' '.join(seed_text)

print(seed_text)
seed_text = 'The  best way to earn a medal is to start working on task and stop'
generateText(model, tokenizer, seqLen[0], seed_text, 25)