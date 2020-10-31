import processing,re 

def get_words():
    sentence = processing.read_sentence()
    sentences = sentence.split(".")
    sentence_words = []*len(sentences)
    for i in range(len(sentences)):
      sentence_words[i].append(re.findall(r'[a-zA-Z]+[-][a-zA-z]+|[a-zA-Z]+[\’]+[a-zA-Z]+|[a-zA-Z]+|[0-9]+',sentences[i]))
    return sentence_words
    

def get_words_expression(sentence):
    return re.findall(r'[a-zA-Z]+[-][a-zA-z]+|[a-zA-Z]+[\’]+[a-zA-Z]+|[a-zA-Z]+|[0-9]+',sentence)

def bigrams():
    words = get_words()
    bigrams = []
    for i in range(len(words)-1):
        bigrams.append([words[i],words[i+1]])
    return bigrams

if __name__ == "__main__":
    scores = processing.read_scores()
    words = get_words()
    bg = bigrams()