#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:52:48 2020

@author: oscargalindo
"""
import model1,feature_extraction, numpy as np

def calculate_values_per_sentence():
    model = model1.get_trained_linear_model()
    d = model1.get_glove()
    sentences = feature_extraction.get_words()
    scores_per_sentence = [[0,0]for i in range(len(sentences))]
    for sentence in range(len(sentences)):
        scores_per_sentence[sentence][-1] = len(sentences[sentence])
        for word in sentences[sentence]:
            if word.lower() in d:
                w = np.array(d[word.lower()])
                scores_per_sentence[sentence][0] += model.predict(w.reshape(1,-1))
    return np.array(scores_per_sentence)

if __name__=="__main__":
    sentences_values = calculate_values_per_sentence()