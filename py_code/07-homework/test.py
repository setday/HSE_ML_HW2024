import os

from sklearn.model_selection import train_test_split

from text_refactorer import refactor_data
from loaders import import_data, import_model, export_model
from model_worker import model_fit, model_predict

def test_data_refactoring():
    data = [
        'Use to love this Airline but the this experience was way below what they use to be and more expensive. The food was poor and although cabin staff were nice very little was offered as far as refreshments. Singapore Airlines has definitely lost what was once considered one of the best Airlines in the world and are now very average.',
        '.,&^%$#@!@#$%^&*()_+',
        'Stop words: a, an, the, in, on, at, to, for, of, with, as, by, and, or, but, not, is, am, I',
    ]
    answer = [
        'use love airline experience way use expensive food poor although cabin staff nice little offer far refreshment singapore airline definitely lose consider one best airline world average',
        '',
        'stop word',
    ]

    assert refactor_data(data) == answer

    try:
        refactor_data('not a list')
    except Exception as e:
        assert str(e) == 'Data must be a list of strings'

    try:
        refactor_data(None)
    except Exception as e:
        assert str(e) == 'Data must be a list of strings'

def test_data_import():
    assert import_data('Not a file', True) == {'rating': [0], 'text': ['Not a file']}
    try:
        import_data('Not a file', False)
    except Exception as e:
        assert str(e) == 'Not a file not exists!'

    data = import_data('../../data/singapore_airlines_reviews.csv', True)
    assert data.iloc[3]['text'] == 'Best Airline in the World Best airline in the world, seats, food, service are all brilliant.  The crew are friendly and welcoming.  We love flying with Singapore Airlines'

    try:
        import_data('../../data/audi.csv', False)
    except Exception as e:
        assert str(e) == 'Wrong data format. No `text` or `rating` columns / fields.'

def test_train_tandem():
    data = import_data('../../data/singapore_airlines_reviews.csv', True)
    data, trash = train_test_split(data, train_size=0.1, random_state=113)
    model, vectorizer = import_model('not_a_file.pkl', True)
    data, test = train_test_split(data, train_size=0.8, random_state=113)

    model_fit(model, vectorizer, data, test)
    export_model(model, vectorizer, 'test_model.pkl')

    assert os.path.exists('test_model.pkl')
    assert os.path.exists('test_model_vectorizer.pkl')

    model, vectorizer = import_model('test_model.pkl', False)

    assert model.classes_ is not None
    assert vectorizer.vocabulary_ is not None

    os.remove('test_model.pkl')
    os.remove('test_model_vectorizer.pkl')
