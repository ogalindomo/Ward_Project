# -*- coding: utf-8 -*-
"""
This file utilizes all the functions from the other files to 
generate and evaluate predictions on the importance of the sentences
"""
import Utilities as util,Evaluation as eval, ScoreModel, LogisticModel


if __name__ == "__main__":
    #Extract the sentences and their values from the dataset
    sentences, values = util.extract_sentences_values()
    sentences, values = util.shuffle_sentences_values(sentences, values)
    
    #Derive the y training and testing data from values
    y_train, y_test = util.train_test_split(values)
    
    #Derive the x training and testing data for the score model
    word_frequencies = ScoreModel.create_word_frequencies(sentences)
    scores_train, scores_test = util.train_test_split(sentences)
    scores_train, scores_test = ScoreModel.generate_scores(scores_train, scores_test, word_frequencies)

    #Save the prediction of the score model for training for the Logistic Model
    score_train_pred = ScoreModel.predict_importance_score(scores_train, y_train, scores_train)
    
    #Get the training and testing data for size features
    size_feat = LogisticModel.get_sizes(sentences)
    size_train, size_test = util.train_test_split(size_feat)
    
    #Get the training and testing data for embedding features
    glove = util.get_glove()
    embeddings = LogisticModel.get_sentence_embedding(sentences, glove)
    embedding_train, embedding_test = util.train_test_split(embeddings)
    
    #Combine the predictions with the size and embedding features to be training
    #for the logistic model
    log_train = LogisticModel.combine(score_train_pred, size_train, embedding_train)
    
    #Make predictions using the logistic model and evaluate them
    log_train_pred = LogisticModel.predict_importance(log_train, y_train, log_train)
    eval.evaluate(log_train_pred, y_train, "Training")
    
    #Setup the testing data for the logistic model
    score_test_pred = ScoreModel.predict_importance_score(scores_train, y_train, scores_test)
    log_test = LogisticModel.combine(score_test_pred, size_test, embedding_test)
    
    #Make predictions using the logistic model and evaluate them
    log_test_pred = LogisticModel.predict_importance(log_train, y_train, log_test)
    eval.evaluate(log_test_pred, y_test, "Testing")
    
    
    
    
    
    
    
    