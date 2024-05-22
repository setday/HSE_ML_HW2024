from sklearn.metrics import f1_score


def model_predict(model, vectorizer, data):
    x_data = vectorizer.transform(data['text'])

    return model.predict(x_data)

def model_fit(model, vectorizer, data, test):
    x_data = vectorizer.fit_transform(data['text'])
    y_data = data['rating']
    
    x_test = vectorizer.transform(test['text'])
    y_test = test['rating']

    model.fit(
        x_data, y_data,
        eval_set=(x_test, y_test),
    )
    test_pred = model_predict(model, vectorizer, test)

    score = f1_score(y_test, test_pred, average='weighted')
    print(f'New score: {score}')
