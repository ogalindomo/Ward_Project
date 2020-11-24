import nltk, csv
from glob import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
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
        sentences = sentence_tokenizer.tokenize(email)
        # sentences = list(map(lambda x: x.replace("\n", " "), sentences))
        
        # print(sentences)

        all_sentences.extend(sentences)

    return all_sentences

def create_word_frequencies(sentences):
    
    # empty dictionary
    word_counts = {}

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        # print(words)

        for word in words:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

    # turn word counts into normalized frequencies
    max_count = np.max(list(word_counts.values()))
    word_frequencies = {}

    for word in word_counts:
        word_frequencies[word] = word_counts[word] / max_count

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
    sentences = sentence_tokenizer.tokenize(email)
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

    # oscar
    # griffith-j
    # stepenovitch-j
    
    # aaron
    # lay-k
    # cash-m

    # isaac
    # king-j
    # maggi-m

    folders = ["griffith-j", "stepenovitch-j", "lay-k", "cash-m", "king-j", "maggi-m"]

    for folder in folders:

        files = glob("./Emails/"+folder+"/*")

        for email in files:

            print(f"tokenizing: {email}\n")
            with open(email, "r", encoding="utf-8") as f:
                with open(email+"scores", "w", encoding="utf-8") as g:
                    data = f.read()

                    writer = csv.writer(g)

                    # print(data)
                    sentences = sentence_tokenizer.tokenize(data)

                    for s in sentences:
                        writer.writerow([s.replace("\n", "")])

def get_training_data():
    emails = glob("./tags/*/*")

    return emails

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

def mse(p,y):
    return np.mean((p-y)**2)

def main():

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
    X_train = sentences[:int(len(sentences)*.9)]
    X_test = sentences[int(len(sentences)*.9):]

    y_train = values[:int(len(sentences)*.9)]
    y_test = values[int(len(sentences)*.9):]

    assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)

    # X_train and X_test are currently arrays of strings
    # we need to turn these into real values by using the frequency scores
    word_frequencies = create_word_frequencies(X_train)
    # print(word_frequencies)

    X_train_real = np.array(list(map(lambda x: score_sentence(x, word_frequencies), X_train))).reshape(-1, 1)
    X_test_real = np.array(list(map(lambda x: score_sentence(x, word_frequencies), X_test))).reshape(-1, 1)
    
    # logistic regression
    model = LogisticRegression()
    model.fit(X_train_real, y_train)

    pred = model.predict(X_test_real)

    print(f"accuracy: {accuracy(pred, y_test)}")


if __name__ == "__main__":
    main()