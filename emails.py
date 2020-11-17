import nltk
from glob import glob
import pandas as pd
import numpy as np

sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
stopwords = nltk.corpus.stopwords.words("english")

def get_email_files():
    return glob("./Emails/*/*")

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
        sentences = list(map(lambda x: x.replace("\n", " "), sentences))
        
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
        score += word_frequencies[word]

    return score

def summarize_email(email, word_frequencies):
    sentences = sentence_tokenizer.tokenize(email)
    sentences = list(map(lambda x: x.replace("\n", " "), sentences))

    scores = list(map(lambda x: score_sentence(x, word_frequencies), sentences))
    print("summarizing email...")
    print(f"email: \n{email}\n")
    print(f"sentence scores: \n{scores}\n")

    max_score = np.argmax(scores)
    print(f"max score: {max_score}")
    print(f"summary: \n{sentences[max_score]}\n")


def main():
    emails = get_all_emails(limit=None)
    sentences = create_sentences(emails)
    # print(sentences)

    word_frequencies = create_word_frequencies(sentences)
    # print(word_frequencies)

    # s = sentences[0]
    # print(f"example sentence: {s}")
    # print(f"sentence score:   {score_sentence(s, word_frequencies)}")

    summarize_email(emails[1], word_frequencies)

    scores = []
    for sentence in sentences:
        scores.append(score_sentence(sentence, word_frequencies))
    
    df = pd.DataFrame({"sentence": sentences, "score": scores})
    # print(df)

    df.to_csv("scores.csv", index=False)


if __name__ == "__main__":
    main()