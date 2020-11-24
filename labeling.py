#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:59:23 2020

@author: oscargalindo
"""


import os 


d = '/Users/oscargalindo/Desktop/Classes/CS 5319/Ward_Project/Tes'

for f in os.listdir(d):
    if f.endswith(".scores"):
        sentences = open(os.path.join(d,f)).readlines()
        for sentence in range(len(sentences)):
            sentences[sentence] = sentences[sentence].strip("\n")
        scores = [0]*len(sentences)
        for sentence in range(len(sentences)):
            print(sentences[sentence])
            matters = input("Matters ?\n")
            matters = int(matters)
            scores[sentence] = matters
        file = open(os.path.join(d,f),"w")
        for sentence in range(len(sentences)):
            file.write(sentences[sentence]+"; "+str(scores[sentence])+";\n")
        file.close()
        print(scores)
        