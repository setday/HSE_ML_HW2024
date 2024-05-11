import os

import pandas as pd
import pickle
from catboost import CatBoostClassifier


random_state = 113

def import_data(data, allowed_inline=False):
    if not os.path.exists(data):
        if allowed_inline:
            return { 'text': data }
        else:
            print(f'{data} not exists!')
            exit(6)
    if data[-4:] != '.csv':
        print(f'{data} has unsupported format. Only `csv` is supported!')
        exit(5)
    
    return pd.read_csv(data)

def import_model(model):
    if not model:
        print(f'No model, no result. All done!')
        exit(4)
    if os.path.exists(model):
        return pickle.loads(model)
    
    return CatBoostClassifier(iterations=1000, loss_function='MultiClass', random_seed=random_state, task_type='GPU', verbose=False)

def export_model(model, path):
    if not path:
        print(f'No model, no result. All done!')
        exit(4)

    pickle.save(model)
