import random

def read_sentence():
    s = "I am Oscar Galindo a graduate student in the Master’s of Computer Science program. My ID is 80585887. I am sending this email to ask you to remove the “Pre-requisite and test score error” requirement to join to the class Topics in Data Science, the CRN of the class is 29030."
    return s
    
def read_scores():
    scores = [0,1,1]
    return scores

def read_scores_words():
    scores = [
        [0.1,0.1,0.3,0.3,0.2,0.2,0.2,0.1,0.1,0.7,0.7,0.3],
        [0.1,0.1,0.8,0.9],
        [0.1,0.2,0.2,0.1,0.1,0.1,0.3,0.4,0.1,0.4,0.1,0.1,0.1,0.1,0.2,
         0.3,0.1,0.1,0.1,0.1,0.3,0.6,0.1,0.2,0.8,0.8,0.1,0.8,0.1,0.8,
         0.7,0.1,0.4,0.1,0.4]
        ]
    return scores

if __name__=="__main__":
    sentence = read_sentence()
    scores_by_words = read_scores()