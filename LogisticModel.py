# -*- coding: utf-8 -*-
"""
This file holds all the functions for the Logistic Regression model that 
uses the Linear Regression Model results along with embeddings and sizes
to predict the importance of each sentence
"""
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression

#Returns a list denoting the size of each sentence
def get_sizes(sentences):
    sizes = []
    for sentence in sentences:
        sizes.append(len(nltk.word_tokenize(sentence)))
    return sizes

#Return a list of average embeddings for every sentence
def get_sentence_embedding(sentences, glove):
    #declare variables
    embeddings = np.zeros((len(sentences),50))
    emb_addition = np.zeros((50))
    for index in range(len(sentences)):
        sentence = sentences[index]
        words = nltk.word_tokenize(sentence)
        
        #Add all the embeddings
        count = 0
        for word in words:
            if word in glove:
                count+=1
                emb_addition += glove[word]
        #Average out the embeddings
        embeddings[index] = emb_addition
        if count != 0:
            embeddings[index] /= count 
        #reset the addtion variable
        emb_addition -= emb_addition
    return embeddings

#Combine the score data, size data, and embeddings to make new training data
def combine(scores, sizes, embeddings):
    result = np.zeros((len(scores), 52))
    for i in range(len(scores)):
         result[i,0], result[i,1], result[i,2:] = scores[i], sizes[i], embeddings[i]
    return result

#Make final predictions for x_test
def predict_importance(x_train, y_train, x_test):
    l_r = LogisticRegression(max_iter=10000)
    l_r.fit(x_train, y_train)
    return l_r.predict(x_test)