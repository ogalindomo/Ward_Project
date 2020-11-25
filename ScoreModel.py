# -*- coding: utf-8 -*-
"""
This file holds all the functions used in the Linear Regression model that 
predicts importance by using the frequency score of each sentence
"""
from sklearn.linear_model import LinearRegression
import nltk
import numpy as np

    
#Compute the frequency for every word that occurs 
def create_word_frequencies(sentences):
    #Empty dictionary
    word_counts = {}

    #Count word Occurences
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

    # Compute maximum value and normalize the values
    max_count = np.max(list(word_counts.values()))
    word_frequencies = {word: value / max_count for word, value in word_counts.items()}
    return word_frequencies



#Give each sentence a score by computing the sum of its word frequencies
def score_sentence(sentence, word_frequencies):
    score = 0
    for word in nltk.word_tokenize(sentence):
        if word in word_frequencies:
            score += word_frequencies[word]
        else:
            score += 0.001
    return score

#Compute the scores for each sentence found in the data
def generate_scores(x_train, x_test, word_frequencies):
    x_train_scores = np.array(list(map(lambda x: score_sentence(x, word_frequencies), x_train))).reshape(-1, 1)
    x_test_scores = np.array(list(map(lambda x: score_sentence(x, word_frequencies), x_test))).reshape(-1, 1)
    return x_train_scores, x_test_scores

#Returns predictions for x_test_scores given the training data
def predict_importance_score(x_train_scores, y_train, x_test_scores):
    model = LinearRegression()
    model.fit(x_train_scores, y_train)
    return model.predict(x_test_scores)