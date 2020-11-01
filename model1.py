import feature_extraction,processing,numpy as np,random
from sklearn.linear_model import LinearRegression,LogisticRegression

def get_glove():
    f = open('glove.6B.50d.txt')
    f = f.readlines()
    d = dict()
    for line in f:
        entries = line.split()
        d[entries[0]] = [float(entries[1+i]) for i in range(len(entries[1:]))]
    return d

def get_scores():
    return processing.read_scores()

def prepare_data():
    x = feature_extraction.get_words() #Vectors for every sentence
    y = processing.read_scores()
    d = get_glove()
    sentence_embedding = np.zeros((len(x),50))
    for sentence in range(len(x)):
        for word in x[sentence]:
            w = word.lower()
            if w in d:
                sentence_embedding[sentence] += d[w]
        sentence_embedding[sentence] /= len(x[sentence])
    return y,sentence_embedding

def get_embeddings():
    l = []
    d = get_glove()
    words = (feature_extraction.get_words())
    for sentence in words:
        for word in sentence:
            if word.lower() in d:
                l.append(d[word.lower()])
            else:
                l.append([0]*50)
    return np.array(l)

def flatten(arr):
    l = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            l.append(arr[i][j])
    return np.array(l)


def get_trained_linear_model():
    x = get_embeddings()
    y = processing.read_scores_words()
    y = flatten(y)
    reg = LinearRegression()
    reg.fit(x,y)
    return reg

if __name__ == "__main__":
    # e = get_embeddings()
    # print(e)
    x,y = get_trained_linear_model()