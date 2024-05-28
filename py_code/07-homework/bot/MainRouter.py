from aiogram import Router, Bot
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.filters import CommandStart

from bot.model.ModelWorker import predict_rating

class MainRouter(Router):
    def __init__(self, bot: Bot) -> None:
        super().__init__()

        self.bot = bot

        self.message.register(self.enter_handler, CommandStart())
        self.message.register(self.default_handler)

    async def default_handler(self, message: Message) -> None:
        rating = predict_rating(message.text)[0][0]
        await message.answer(f"Даю этому тексту оценку: {rating}")
    
    async def enter_handler(self, message: Message, state: FSMContext) -> None:
        await message.answer("Привет! Я бот который даёт оценку комментариям.\n\nНапиши мне что-нибудь, и я попробую оценить его.")
