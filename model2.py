#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:21:57 2020

@author: oscargalindo
"""
from sklearn.linear_model import LogisticRegression
import model1,processing,values, numpy as np

def get_trained_logistic_model(): #sentences through embeddings
    y,x = model1.prepare_data()
    reg = LogisticRegression()
    reg.fit(x,y)
    return reg

def get_trained_logistic():
    y = processing.read_scores()
    x = values.calculate_values_per_sentence()
    reg = LogisticRegression()
    reg.fit(x,y)
    return reg 

if __name__=="__main__":
    # model = get_trained_logistic_model()
    # y,x = model1.prepare_data()
    # print(model.predict(x[1].reshape(1,-1)))
    
    model = get_trained_logistic()
    x = values.calculate_values_per_sentence()
    print(model.predict(x[2].reshape(1,-1)))