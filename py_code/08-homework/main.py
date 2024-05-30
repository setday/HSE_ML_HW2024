import click

from loaders import import_data, import_model, export_model
from model_worker import model_fit, model_predict


@click.command()
@click.argument('file_to_detect')
def main(file_to_detect):    
    if not file_to_detect:
        raise ValueError(f'No file to detect language')
    data = import_data(file_to_detect)

    model = import_model('model/model_recognize_language.pkl', 'model/vectorizer_recognize_language.pkl')

    predict_data = model_predict(model, data)
    predict_data = predict_data[0]

    real_labels = {
        0: 'Just text',
        1: 'Python',
        2: 'Cpp',
        3: 'Js',
        4: 'Java',
        5: 'Yaml',
        6: 'Bash',
        7: 'Markdown',
        8: 'C',
        9: 'Kotlin',
        10: 'Haskell',
    }

    print(real_labels[predict_data])


if __name__ == '__main__':
    main()
