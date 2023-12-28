import re

from models.model import Model
from inference.errors import ParseError, InternalError, WrongParamsError


def parse_input(features: dict, message: str) -> dict:
    """Парсинг входящего сообщения.

    Args:
        features (dict): словарь, описывающий параметры модели; Model.get_features_dict()
        message (str): сообщение;

    Raises:
        ParseError: если мессендж не соответствует формату ввода;

    Returns:
        dict: содержащий параметры на вход модели.
    """
    try:
        message = re.sub("^\s+|\n|\r|\s+$", "", message)  # noqa: W605
        message = message.replace(" ", "")

        params = {}
        for pair in message.split(","):
            if not pair or pair == '':
                continue

            key, value = pair.split(":", 1)
            if key in params:
                raise Exception
            if key not in features:
                continue
            if features[key] == "numeric":
                params[key] = float(value)
            else:
                params[key] = str(value).strip()

        return params
    except Exception:
        raise ParseError


def prepare_output(result) -> str:
    """Возвращает сообщение с ответом инференса;

    Args:
        result (_type_): результат выполнения инференса;

    Returns:
        str: строка с ответом.
    """
    if result[0][0] == 'lived':
        return f'Рекомендация: лечение будет эффективным с вероятностью {result[1] * 100: .2f}%.'

    return f'Рекомендация: лечение не эффективно с вероятностью {result[1] * 100: .2f}%.'


def process(model: Model, message: str) -> str:
    """Запуск инференса модели;

    Args:
        model (Model): модель;
        message (str): необработанное входное сообщение;

    Returns:
        str: сообщение-ответ.
    """
    try:
        data = parse_input(model.get_features_dict(), message)
        result = model.predict(data)
        return prepare_output(result)
    except (ParseError, WrongParamsError):
        return "Возникла ошибка парсинга, либо введены неверные параметры."
    except InternalError:
        return "Внутренняя ошибка. Обратитесь к администраторам."
