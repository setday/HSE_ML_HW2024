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
            raise FileNotFoundError(f'{data} not exists!')
    if data[-4:] != '.csv':
        raise ValueError(f'{data} has unsupported format. Only `csv` is supported!')
    
    out_data = pd.read_csv(data)
    if 'text' not in out_data.columns:
        raise ValueError(f'Wrong data format. No `text` or `rating` columns / fields.')
    
    out_data['rating'] = out_data['rating'].astype(int) if 'rating' in out_data.columns else 0
    out_data['text'] = out_data['text'].astype(str)
    
    if 'title' in out_data.columns:
        out_data['title'] = out_data['title'].astype(str)
        out_data['text'] = out_data['title'] + ' ' + out_data['text']
    
    return out_data[['text', 'rating']]

def import_model(model_name, allowed_new=False):
    if not model_name:
        raise ValueError(f'No model, no result. All done!')
    if os.path.exists(model_name) and os.path.exists(model_name[:-4] + '_vectorizer.pkl'):
        with open(model_name, 'rb') as file:
            model = pickle.load(file)
        with open(model_name[:-4] + '_vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        return (
            model,
            vectorizer
        )
    
    if allowed_new:
        return (
            CatBoostClassifier(iterations=1000, loss_function='MultiClass', random_seed=random_state, task_type='GPU', verbose=False),
            TfidfVectorizer()
        )
    else:
        raise FileNotFoundError(f'{model_name} or {model_name[:-4] + "_vectorizer.pkl"} not exists!')

def export_model(model, vectorizer, path):
    if not path:
        raise ValueError(f'Can\'t export model to empty path!')
        
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    with open(path[:-4] + '_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
