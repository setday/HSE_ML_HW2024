from loaders import import_model
from model_worker import model_predict
from text_refactorer import refactor_data

model, vectorizer = import_model("bot/model/model.pkl")

def predict_rating(text: str) -> int:
    return model_predict(model, vectorizer, { 'text': text })
