import feature_extraction,processing,numpy as np,random
from sklearn.linear_model import LinearRegression

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
  
    x = []*len()

def train_model():
    y,x = prepare_data()
    reg = LinearRegression()
    reg.fit(x,y)
    return reg

if __name__=="__main__":
    model = train_model()