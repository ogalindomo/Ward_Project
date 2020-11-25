# -*- coding: utf-8 -*-
"""
This file holds various "utility" functions that do not belong to a particular
model or process
"""

from glob import glob
import numpy as np
gloveDirectory = "../../glove.6B.50d.txt"

#Retrieve Glove Embeddings in a Dictionary Format
def get_glove():
    with open(gloveDirectory, encoding = "utf-8") as gloveFile:
        contents = gloveFile.readlines()
        embeddings = dict()
        for line in contents:
            entries = line.split()
            embeddings[entries[0]] = [float(entries[1+i]) for i in range(len(entries[1:]))]
    return embeddings

#Returns numpy arrays of all the sentences in the data and their ratings
def extract_sentences_values():
    emails = glob("./tags/*/*")
    scores = {}
    for email in emails:
        with open(email, "r", encoding="utf-8") as file:
            text = file.read()
            lines = text.splitlines()

            for line in lines:
                data = line.split(";")[:2]
                #Check for validity
                if len(data) == 2 and len(data[1].strip()) == 1:
                    scores[data[0].strip('"').replace("\n", "")] = int(data[1].strip())

    #Store the sentences and values
    sentences = list(scores.keys())
    values = list(scores.values())
    
    return sentences, values

#Randomize order of the data
def shuffle_sentences_values(sentences, values):
    order = np.random.permutation(len(sentences))
    sentences = np.asarray(sentences)[order]
    values = np.asarray(values)[order]
    return sentences, values

#Split the given x into training and test data
def train_test_split(x, fraction = .8):
    #get the index to split on
    index = int(len(x) * fraction)
    
    # split into train and test data
    x_train = x[:index]
    x_test = x[index:]
    
    return x_train, x_test
    
