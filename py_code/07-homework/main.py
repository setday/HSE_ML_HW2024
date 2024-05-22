import click

from sklearn.model_selection import train_test_split

from loaders import import_data, import_model, export_model
from model_worker import model_fit, model_predict
from text_refactorer import refactor_data


random_state = 113

@click.command()
@click.option('--data', required=False, help='Data sheet to be trained on or to be predicted.', type=str)
@click.option('--test', required=False, help='Test data sheet. `predict` mode only! ', type=str)
@click.option('--model', required=False, help='Model in pickle format path.', type=str)
@click.option('--split', required=False, help='Test split proportion. `predict` mode only! [default: 0.8]', type=float)
@click.argument('command')
def main(command, data, test, model, split):
    if command not in ['train', 'predict']:
        raise ValueError(f'Invalide argument {command}. Expected `train` or `predict`!')
    if command == 'predict' and (test or split):
        raise ValueError(f'--test and --split can\'t be used in predict mode!')
    if test and split:
        raise ValueError(f'--test and --split can\'t be used simultaneously!')
    if not test and not split:
        split = 0.8
    
    if not data:
        raise ValueError(f'No data to work with!')
    data = import_data(data, command == 'predict')
    
    if test:
        test = import_data(test)
    elif command == 'train':
        data, test = train_test_split(data, train_size=split, random_state=random_state)

    model_name = model
    model, vectorizer = import_model(model_name, command == 'train')

    if command == 'predict':
        predict_data = model_predict(model, vectorizer, data)
        predict_data = [el[0] for el in predict_data]
        print(*predict_data, sep='\n')
    else:
        model_fit(model, vectorizer, data, test)
        export_model(model, vectorizer, model_name)


if __name__ == '__main__':
    main()
