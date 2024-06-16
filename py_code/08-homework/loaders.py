import os

import pickle


def import_data(data):
    if not os.path.exists(data):
        return data
    with open(data, 'r') as file:
        data = file.read()
    return data

def import_model(model_name, vectorizer_name):
    try:
        with open(model_name, 'rb') as file:
            model = pickle.load(file)
        with open(vectorizer_name, 'rb') as file:
            vectorizer = pickle.load(file)
        return (
            model,
            vectorizer
        )
    except Exception as e:
        raise ValueError(f'Error while loading model: {e}')
