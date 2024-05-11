from sklearn.metrics import f1_score


def model_predict(model, data):
    x_data = data['text']

    return model.predict(x_data)

def model_fit(model, data, test):
    x_data = data['text']
    y_data = data['rating']
    
    x_test = test['text']
    y_test = test['rating']

    print(x_data, y_data)

    model.fit(
        x_data, y_data,
    )
    test_pred = model_predict(model, test)

    score = f1_score(y_test, test_pred, average='weighted')
    print(f'New score: {score}')
