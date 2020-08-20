# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 23:34:54 2020

@author: Tanvy
"""


import pickle
import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer 

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM
from keras.layers.embeddings import Embedding

with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)
    
with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)


allData = test_data + train_data

print(len(allData))

vocab = set()

for story, ques, ans in allData:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(ques))
    
vocab.add('yes')
vocab.add('no')

print(vocab)

vocabLen = len(vocab) + 1
print(vocabLen)

allStoryLen = [len(data[0]) for data in allData]
print(allStoryLen)

maxStoryLen  = max(allStoryLen)

allQuesLen = [len(data[1]) for data in allData]
print(allQuesLen)

maxQuesLen  = max(allQuesLen)

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

print(len(tokenizer.index_word))

train_story = []
train_ques  = []
train_ans   = []

for story, ques, ans in train_data:
    train_story.append(story)
    train_ques.append(ques)
    train_ans.append(ans)
    
train_story_seq = tokenizer.texts_to_sequences(train_story)

def vectorizeStories(data, word_index=tokenizer.word_index, max_story_len=maxStoryLen, max_ques_len = maxQuesLen):
    X  = []
    Xq = []
    Y  = []
    
    for story, ques, ans in data:
        x  = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in ques]
        
        y = np.zeros(len(word_index) + 1)
        y[word_index[ans]] = 1
        
        X.append(x)
        Y.append(y)
        Xq.append(xq)

    return (pad_sequences(X, maxlen=maxStoryLen), pad_sequences(Y, maxlen=maxQuesLen), np.array(Y))

input_test, query_test, ans_test    = vectorizeStories(train_data)
input_train, query_train, ans_train = vectorizeStories(test_data)

input_seq = Input((maxStoryLen,))
questions = Input((maxQuesLen,))

vocab_size = len(vocab) + 1

input_seq_m = Sequential()
input_seq_m.add(Embedding(input_dim=vocab_size, output_dim=64))
input_seq_m.add(Dropout(0.3))

input_seq_c = Sequential()
input_seq_c.add(Embedding(input_dim=vocab_size, output_dim=maxQuesLen))
input_seq_c.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=maxQuesLen))
question_encoder.add(Dropout(0.3))

input_encoded_m  = input_seq_m(input_seq)
input_encoded_c  = input_seq_c(input_seq)
question_encoded = question_encoder(questions)

match = dot([input_encoded_m, question_encoded], axes=(2,2))
match = Activation("softmax")(match)

response = add([match, input_encoded_c])
response = Permute((2,1))(response)

answer   = concatenate([response, question_encoded])

print(answer)


answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)

answer = Activation('softmax')(answer)

model  = Model([input_seq, questions], answer)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit([input_train, query_train], ans_train, batch_size=32, epochs=3, validation_data=([input_test, query_test], ans_test))

pred_res = model.predict(([input_test, query_test]))

print(pred_res[0])

val_max = np.argmax(pred_res[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key
        
print(k)

myStory = "Sandra dropped the football in garden"
myQues  = "Is football in garden ?"

myData  = [(myStory.split(), myQues.split(), "yes")]

myStory, myQues, myAns = vectorizeStories(myData)

myRes    = model.predict(([myStory, myQues]))

val_max = np.argmax(myRes[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key
        
print(k)