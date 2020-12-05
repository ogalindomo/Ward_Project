# -*- coding: utf-8 -*-
"""
This file holds all the functions used to evaluate the results of the program
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix 
import matplotlib.pyplot as plt

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

def mse(p,y):
    return np.mean((p-y)**2)

def compute_fraction(vector):
    vector = np.asarray(vector)
    return np.sum(vector == 0)/len(vector)

def evaluate(pred, act, event):
    print("Here are the {} data results:".format(event))
    print(f"percentage of 0's: {compute_fraction(act)}")
    print(f"percentage of 1's: {1 - compute_fraction(act)}")
    print(f"accuracy: {accuracy(pred, act)}")
    print(f"mse: {mse(pred, act)}")
    print(f"precision: {precision_score(pred, act)}")
    print(f"recall: {recall_score(pred, act)}")
    print(f"f1: {f1_score(pred, act)}")
    print("confusion matrix:")
    conf_matrix = confusion_matrix(pred, act)
    print(f"{conf_matrix}")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title('{} Data Confusion Matrix'.format(event), fontsize=18)
    plt.savefig('confusion_matrix_test.png')
    print("\n\n")
