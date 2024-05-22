import string
from typing import List

import nltk
import pandas as pd


# nltk.download('wordnet')
# nltk.download("omw-1.4")
# nltk.download('stopwords')

stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
translator = str.maketrans('', '', string.punctuation)

def pre_process(text):
    text = text.lower()
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

def prefactoring(text, title=None):
    if text is None or not isinstance(text, str):
        raise ValueError('Text must be a string')
    if title and not isinstance(title, str):
        raise ValueError('Title must be a string')

    if title:
        text = title + ' ' + text
    text = pre_process(text)
    text = remorph(text)
    text = remove_stopwords(text)
    return text

def refactor_data(data: List[str] | pd.Series) -> List[str]:
    if isinstance(data, pd.Series):
        data = data.tolist()
    if data is None or not isinstance(data, list) or not all(isinstance(review, str) for review in data):
        raise ValueError('Data must be a list of strings')

    return [
        prefactoring(review)
        for review in data
    ]
