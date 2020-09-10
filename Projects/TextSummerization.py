# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:16:27 2020

@author: Tanvy
"""


import spacy
from string import punctuation
from collections import Counter

nlp = spacy.load('en_core_web_lg')


def top_sentence(text, limit):
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    
    doc     = nlp(text.lower())
    
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        
        if token.pos_ in pos_tag:
            keyword.append(token.text)
            
    freq_word = Counter(keyword)
    max_word  = Counter(keyword).most_common(1)[0][1]
    
    for w in freq_word:
        freq_word[w] = freq_word[w]/max_word
        
    sent_strength = {}
    
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent]  = freq_word[word.text]
                    
                    
    summary  = []
    
    sorted_x = sorted(sent_strength.items(), key=lambda kv:kv[1], reverse=True)
    
    counter = 0
    
    for i in range(len(sorted_x)):
        summary.append(str(sorted_x[i][0]))
        counter += 1
        if counter >= limit:
            break
        
    return  ' '.join(summary)

#sampleTxt = '''Yamaha is reminding people that musical equipment cases are for musical equipment — not people — two weeks after fugitive auto titan Carlos Ghosn reportedly was smuggled out of Japan in one. In a tweet over the weekend, the Japanese musical equipment company said it was not naming any names, but noted there had been many recent stories about people getting into musical equipment cases. Yamaha (YAMCY) warned people not to get into, or let others get into, its cases to avoid "unfortunate accidents." Multiple media outlets have reported that Ghosn managed to sneak through a Japanese airport to a private jet that whisked him out of the country by hiding in a large, black music equipment case with breathing holes drilled in the bottom. CNN Business has not independently confirmed those details of his escape. The former Nissan (NSANF) CEO had been out on bail awaiting trial in Japan on charges of financial wrongdoing before making his stunning escape to Lebanon at the end of December. Ghosn has referred to his departure as an effort to "escape injustice." In an interview with CNN\'s Richard Quest last week, Ghosn did not comment on the nature of his escape, saying he didn\'t want to endanger any of the people who aided in the operation. Ghosn did, however, respond to a question about what it felt like to ride through the airport in a packing case by first declining to comment but then adding: "Freedom, no matter the way it happens, is always sweet." In a press conference in Lebanon ahead of the CNN interview last Wednesday, Ghosn\'s first public appearance since fleeing Japan, Ghosn said he decided to leave the country because he believed he would not receive a fair trial, a claim Japanese authorities have disputed. Brands sometimes capitalize on their tangential relationship to big news in order to attract attention on social media. Yamaha is one of Japan\'s best known brands and Ghosn was one of Japan\'s top executives before being ousted from Nissan — a match made in social media heaven. Not surprisingly, Yamaha\'s post went viral on Twitter over the weekend.'''

sampleTxt = ''' Store highlights is a summary created for the bigger article. News data from CNN and Daily Mail was collected to create the CNN/Daily Mail data set for text summarization which is the key data set used for training abstractive summarization models. Using this data set as benchmark, researchers have been experimenting with deep learning model designs.
One such model that I love is the Pointer Generator Network by Abigail See. I want to use this model to highlight the key components of a deep learning summarization model.
Before we get to the model lets talk about the metrics for evaluation of text summarization — Rouge Score. Rouge score highlights the word overlap between the summarized and the source text. Rouge 1 — measures single word overlap between source and summarized text whereas Rouge 2 measures bi gram overlap between source and summary. Since rouge score metric only looks at word overlap and not readability of the text it is not a perfect metric as text with high rouge score can be a badly written summary. '''

print(top_sentence(sampleTxt, 8))