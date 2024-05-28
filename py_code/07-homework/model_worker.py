from sklearn.metrics import f1_score
from text_refactorer import refactor_data


def model_predict(model, vectorizer, data):
    if model is None or vectorizer is None or data is None:
        raise ValueError(f'Can\'t predict without model / vectorizer / data.')
    if 'text' not in data:
        raise ValueError(f'Wrong data format. No `text` column / field.')
    if vectorizer.vocabulary_ is None:
        raise ValueError(f'Vectorizer is not fitted. Fit vectorizer first.')
    if model.classes_ is None:
        raise ValueError(f'Model is not fitted. Fit model first.')

    x_data = refactor_data(data['text'])
    x_data = vectorizer.transform(x_data)

    return model.predict(x_data)

def model_fit(model, vectorizer, data, test):
    if model is None or vectorizer is None or data is None or test is None:
        raise ValueError(f'Can\'t fit without model / vectorizer / data / test.')
    if 'text' not in data or 'rating' not in data:
        raise ValueError(f'Wrong data format. No `text` or `rating` columns / fields.')
    if 'text' not in test or 'rating' not in test:
        raise ValueError(f'Wrong test format. No `text` or `rating` columns / fields.')

    x_data = refactor_data(data['text'])
    x_data = vectorizer.fit_transform(x_data)
    y_data = data['rating']
    
    x_test = refactor_data(test['text'])
    x_test = vectorizer.transform(x_test)
    y_test = test['rating']

    model.fit(
        x_data, y_data,
        eval_set=(x_test, y_test),
    )
    test_pred = model_predict(model, vectorizer, test)

    score = f1_score(y_test, test_pred, average='weighted')
    print(f'New score: {score}')
