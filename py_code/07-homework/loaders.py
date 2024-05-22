import os

import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


random_state = 113

def import_data(data, allowed_inline=False):
    if not os.path.exists(data):
        if allowed_inline:
            return { 'text': [data], 'rating': [0] }
        else:
            print(f'{data} not exists!')
            exit(6)
    if data[-4:] != '.csv':
        print(f'{data} has unsupported format. Only `csv` is supported!')
        exit(5)
    
    out_data = pd.read_csv(data)
    out_data['rating'] = out_data['rating'].astype(int) if 'rating' in out_data.columns else 0
    out_data['text'] = out_data['text'].astype(str)
    if 'title' in out_data.columns:
        out_data['title'] = out_data['title'].astype(str)
        out_data['text'] = out_data['title'] + ' ' + out_data['text']
    return out_data[['text', 'rating']]

def import_model(model_name):
    if not model_name:
        print(f'No model, no result. All done!')
        exit(4)
    if os.path.exists(model_name):
        with open(model_name, 'rb') as file:
            model = pickle.load(file)
        with open(model_name[:-4] + '_vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        return (
            model,
            vectorizer
        )
    
    return (
        CatBoostClassifier(iterations=1000, loss_function='MultiClass', random_seed=random_state, task_type='GPU', verbose=False),
        TfidfVectorizer()
    )

def export_model(model, vectorizer, path):
    if not path:
        print(f'No model, no result. All done!')
        exit(4)
        
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    with open(path[:-4] + '_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
