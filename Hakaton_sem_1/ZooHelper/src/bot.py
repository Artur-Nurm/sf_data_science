import os
import pickle

from dotenv import load_dotenv
import telebot

from inference.model_inference import process
from models.model import Model
from inference.docs import (
    HELP_DOC,
    model_features,
    PROCESS_DOC,
    REQUEST_HELP,
    HELLO,
    model_features_lesion,
)

# Подгружаем параметры из .env-файла
load_dotenv()
BOT_TOKEN = os.environ.get("BOT_TOKEN")
MODEL_PATH = os.environ.get("MODEL_PATH")

dirname = os.path.dirname(__file__)
dirname = os.path.split(dirname)[0]
model_path = os.path.join(dirname, MODEL_PATH)

# Загружаем модель
xgb = pickle.load(open(model_path, "rb"))
model = Model(xgb)


# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN)


# Далее идёт описание комманд чата
@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, HELLO)


@bot.message_handler(commands=["help"])
def help(message):
    bot.reply_to(message, HELP_DOC)


@bot.message_handler(commands=["features"])
def features(message):
    bot.reply_to(message, model_features(model))


@bot.message_handler(commands=["features_lesion"])
def feaures_lesion(message):
    bot.reply_to(message, model_features_lesion(model))


@bot.message_handler(commands=["process"])
def process_handler(message):
    send = bot.send_message(
        message.chat.id,
        PROCESS_DOC,
    )
    bot.register_next_step_handler(send, process_following)


def process_following(message):
    bot.send_message(message.chat.id, process(model, message.text))


# Если в сообщение не содержится комманда
@bot.message_handler()
def other(message):
    bot.reply_to(message, REQUEST_HELP)


# Запуск
print("Bot started")
bot.infinity_polling()
