import nltk, csv
from glob import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from nltk.tokenize import PunktSentenceTokenizer

gloveDirectory = "../../glove.6B.50d.txt"
#nltk.download('punkt')
#sentence_tokenizer = nltk.data.load("/Library/Frameworks/Python.framework/Versions/3.8/lib/nltk_data/tokenizers/punkt/english.pickle")
stopwords = nltk.corpus.stopwords.words("english")

def get_email_files():
    return glob("./Emails/*/*.")

def get_random_email():
    email_list = get_email_files()

    random_email = np.random.randint(0, len(email_list)-1)

    with open(email_list[random_email], "r", encoding="utf-8") as f:
        data = f.read()
    return data

def get_all_emails(limit=None):
    email_list = get_email_files()

    email_data = []

    if limit:
        n = limit
    else:
        n = len(email_list)
    
    for email in email_list[:n]:

        with open(email, "r", encoding="utf-8") as f:
            email_data.append(f.read())

    return email_data
        
def create_sentences(emails):

    all_sentences = []

    for email in emails:
        # print(f"tokenizing: {email}")

        # TODO: 
        # clean up the sentences with some regex
        sentences = nltk.tokenize(email)
        # sentences = list(map(lambda x: x.replace("\n", " "), sentences))
        
        # print(sentences)

        all_sentences.extend(sentences)

    return all_sentences

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

def score_sentence(sentence, word_frequencies):
    score = 0
    for word in nltk.word_tokenize(sentence):
        if word in word_frequencies:
            score += word_frequencies[word]
        else:
            score += 0.001
    return score

def summarize_email(email, word_frequencies):
    sentences = nltk.tokenize(email)
    # sentences = list(map(lambda x: x.replace("\n", " "), sentences))

    scores = list(map(lambda x: score_sentence(x, word_frequencies), sentences))
    print("summarizing email...")
    print(f"email: \n{email}\n")
    print(f"sentence scores: \n{scores}\n")

    max_score = np.argmax(scores)
    print(f"max score: {max_score}")
    print(f"summary: \n{sentences[max_score]}\n")

    return sentences[max_score]

def tokenize_emails():
    folders = glob("./Emails/*")
    folders = ["griffith-j", "stepenovitch-j", "lay-k", "cash-m", "king-j", "maggi-m"]

    for folder in folders:

        files = glob("./Emails/"+folder+"/*")

        for email in files:

            print(f"tokenizing: {email}\n")
            with open(email, "r", encoding="utf-8") as f:
                with open(email+"scores", "w", encoding="utf-8") as g:
                    data = f.read()

                    writer = csv.writer(g)
                    sentences = nltk.tokenize(data)

                    for s in sentences:
                        writer.writerow([s.replace("\n", "")])

def get_training_data():
    emails = glob("./tags/*/*")

    return emails

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

def mse(p,y):
    return np.mean((p-y)**2)

def get_glove():
    with open(gloveDirectory, encoding = "utf-8") as gloveFile:
        contents = gloveFile.readlines()
        embeddings = dict()
        for line in contents:
            entries = line.split()
            embeddings[entries[0]] = [float(entries[1+i]) for i in range(len(entries[1:]))]
    return embeddings

def embeddings_scores_model(word_scores):
    glove = get_glove()
    model = LogisticRegression()
    y = []
    x = []
    for word in word_scores.keys():
        if word in glove:
            y.append(word_scores[word])
            x.append(glove[word])
    print(x)
    model.fit(x,y)
    return model

def get_sizes(vector):
    v = []
    for element in range(len(vector)):
        v.append(len(nltk.word_tokenize(vector[element])))
    return v

def get_sentence_embedding(vector):
    v = np.zeros((len(vector),50))
    v_addition = np.zeros((50))
    glove = get_glove()
    for element in range(len(vector)):
        words = nltk.word_tokenize(vector[element])
        c = 1
        for word in words:
            if word in glove:
                c+=1
                v_addition += glove[word]
        v[element] = v_addition
        v[element] /= c
        v_addition = np.zeros(50)
    return v

def combine(vector1, vector2):
    result = np.zeros((len(vector1), 2))
    for i in range(len(vector1)):
         result[i,0], result[i,1] = vector1[i], vector2[i]
    return result

def combine3(vector1, vector2, vector3):
    result = np.zeros((len(vector1), 52))
    for i in range(len(vector1)):
         result[i,0], result[i,1], result[i,2:] = vector1[i], vector2[i],vector3[i]
    return result

def print_fraction(vector):
    c_0 = 0
    c_1 = 0
    for e in vector: 
        if e == 0: 
            c_0+=1
        else: 
            c_1+=1
    print("Fraction of zero:",(c_0/len(vector)),"Fraction of one:",(c_1/len(vector)))

def train_scores_model():


    X_train_real = np.array(list(map(lambda x: score_sentence(x, word_frequencies), X_train))).reshape(-1, 1)
    X_test_real = np.array(list(map(lambda x: score_sentence(x, word_frequencies), X_test))).reshape(-1, 1)
    
    # logistic regression
    model = LinearRegression()
    model.fit(X_train_real, y_train)
    pred = model.predict(X_train_real)
    ##########
    sizes = get_sizes(X_train_real)
    new_train = combine(pred, sizes)
    print(new_train)
    l_r = LogisticRegression()
    l_r.fit(new_train, y_train)
    
    
    r = combine(model.predict(X_test_real), get_sizes(X_test_real))
    
    pred = l_r.predict(r)
    

if __name__ == "__main__":
    # scores_model,embeddings_model = train_scores_model()
    

    emails = get_training_data()

    scores = {}

    # preprocessing
    for email in emails:
        with open(email, "r", encoding="utf-8") as f:
            text = f.read()

            lines = text.splitlines()

            for line in lines:
                data = line.split(";")[:2]
                if len(data) == 2 and len(data[1].strip()) == 1:
                    scores[data[0].strip('"').replace("\n", "")] = int(data[1].strip())

    # put the sentences and the scores in separate arrays
    sentences = list(scores.keys())
    values = list(scores.values())

    # split into train and test data
    order = np.random.permutation(len(sentences))
    sentences = np.asarray(sentences)
    values = np.asarray(values)
    sentences = sentences[order]
    values = values[order]
    
    X_train = sentences[:int(len(sentences)*.8)]
    X_test = sentences[int(len(sentences)*.8):] 

    y_train = values[:int(len(sentences)*.8)]
    y_test = values[int(len(sentences)*.8):]

    assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)

    # X_train and X_test are currently arrays of strings
    # we need to turn these into real values by using the frequency scores
    word_frequencies = create_word_frequencies(X_train)
    
    # embeddings_model = embeddings_scores_model(word_frequencies)
    
    # print(word_frequencies)

    X_train_real = np.array(list(map(lambda x: score_sentence(x, word_frequencies), X_train))).reshape(-1, 1)
    X_test_real = np.array(list(map(lambda x: score_sentence(x, word_frequencies), X_test))).reshape(-1, 1)
    
    # logistic regression
    model = LinearRegression()
    model.fit(X_train_real, y_train)
    pred = model.predict(X_train_real)
    ##########
    sizes = get_sizes(X_train)
    e = get_sentence_embedding(X_train)
    new_train = combine3(pred, sizes,e)
    # print(new_train)
    l_r = LogisticRegression(max_iter=10000)
    l_r.fit(new_train, y_train)
    pred = l_r.predict(new_train)
    print_fraction(y_train)
    print(f"accuracy: {accuracy(pred, y_train)}")
    
    e = get_sentence_embedding(X_test)
    
    r = combine3(model.predict(X_test_real), get_sizes(X_test),e)
    
    pred = l_r.predict(r)
    
    print_fraction(y_test)
    print(f"accuracy: {accuracy(pred, y_test)}")    
    