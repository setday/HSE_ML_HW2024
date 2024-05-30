def model_predict(model, data: str):
    model_lr, model_vec = model

    if model_lr is None or model_vec is None or data is None:
        raise ValueError(f'Can\'t predict without model / vectorizer / data.')
    if model_vec.vocabulary_ is None:
        raise ValueError(f'Vectorizer is not fitted. Fit vectorizer first.')

    x_data = model_vec.transform([data])

    return model_lr.predict(x_data)
