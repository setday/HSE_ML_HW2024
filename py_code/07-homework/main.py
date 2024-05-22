import click

from sklearn.model_selection import train_test_split

from loaders import import_data, import_model, export_model
from model_worker import model_fit, model_predict
from text_refactorer import refactor_data


random_state = 113

@click.command()
@click.option('--data', required=True, help='Data sheet to be trained on or to be predicted.', type=str)
@click.option('--test', required=False, help='Test data sheet. `predict` mode only! ', type=str)
@click.option('--model', required=False, help='Model in pickle format path.', type=str)
@click.option('--split', required=False, help='Test split proportion. `predict` mode only! [default: 0.8]', type=float)
@click.argument('command')
def main(command, data, test, model, split):
    if command == 'predict' and (test or split):
        print(f'--split or --test can be used in train mode only!')
        exit(2)
    if test and split:
        print(f'Invalide option combination. --split and --test can\'t be set simultaneously!')
        exit(3)
    if not test and not split:
        split = 0.8

    data = import_data(data, command == 'predict')
    data['text'] = refactor_data(data)
    
    if test:
        test = import_data(test)
    elif command == 'train':
        data, test = train_test_split(data, train_size=split, random_state=random_state)

    model_name = model
    model, vectorizer = import_model(model_name)

    if command == 'predict':
        predict_data = model_predict(model, vectorizer, data)
        predict_data = [el[0] for el in predict_data]
        print(*predict_data, sep='\n')
    elif command == 'train':
        model_fit(model, vectorizer, data, test)
        export_model(model, vectorizer, model_name)
    else:
        print(f'Invalide argument {command}. Expected `train` or `predict`!')
        exit(1)


if __name__ == '__main__':
    main()
