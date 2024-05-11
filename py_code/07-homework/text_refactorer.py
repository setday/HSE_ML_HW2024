import string

import nltk


nltk.download('wordnet')
nltk.download("omw-1.4")
nltk.download('stopwords')

stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

def pre_process(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = ' '.join(text.split())
    return text

def lemmatize(word):
    word = lemmatizer.lemmatize(word, pos="v")
    word = lemmatizer.lemmatize(word, pos="a")
    word = lemmatizer.lemmatize(word, pos="r")
    word = lemmatizer.lemmatize(word, pos="n")
    return word

def remorph(text):
    return ' '.join([lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords])

def prefactoring(text, title):
    text = title + ' ' + text
    text = pre_process(text)
    text = remorph(text)
    text = remove_stopwords(text)
    return text

prefactoring(review['text'], review['title'])
