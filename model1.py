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

def train_model():
    y,x = prepare_data()
    reg = LinearRegression()
    reg.fit(x,y)
    return reg

def train_model_logistic():
    y,x = prepare_data()
    reg = LogisticRegression()
    reg.fit(x,y)
    return reg

if __name__=="__main__":
    model = train_model_logistic()
    y,x = prepare_data()
    print(model.predict(x[1].reshape(1,-1)))