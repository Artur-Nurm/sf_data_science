import inspect

from models.model import Model


HELP_DOC = inspect.cleandoc(
    """
    /process -- ввод данных для расчета;
    /features -- вывод словаря с описанием возможных значений;
    /features_lesion -- описание возможных значений для параметра lesion_1;

    Подробное описание параметров ввода:
    https://github.com/angr1it/zoohelper/blob/main/notebooks/features.md
    """
)

PROCESS_DOC = inspect.cleandoc(
    """
    Введите переменные в формате key: value через запятую.
    Например:
    surgery: yes,
    age: adult,
    rectal_temp: 38.1,
    pulse: 132.0,
    pain: severe_pain
    """
)

REQUEST_HELP = inspect.cleandoc(
    """
    Не знаю такой команды. Попробуйте /help, чтобы посмотреть формат ввода.
    """
)

HELLO = inspect.cleandoc(
    """
    Привет. Это бот решения проблемы предсказания здоровья лошади.
    https://www.kaggle.com/competitions/playground-series-s3e22
    Попробуйте /help, чтобы посмотреть возможные команды.
    """
)


def model_features(model: Model) -> str:
    """Представление словаря с описанием параметров модели в читаемом виде;

    Args:
        model (Model): текущая модель, поддерживает Model.get_features_dict();

    Returns:
        str: строка-сообщение с кратким описанием параметров модели.
    """
    d = model.get_features_dict()

    nums = []
    cats = {}
    yes_no = []

    for key, value in d.items():
        if value[0] == "numeric":
            nums.append(key)
            continue
        if value[0] == "no" or value[0] == "yes":
            yes_no.append(key)
            continue
        if key == "lesion_1":
            continue

        cats[key] = value

    NUMS_STR = f"Numeric features:\n{', '.join(nums)}"
    YES_NO_STR = f"yes/no features:\n{', '.join(yes_no)}"
    CATS_STR = "Categorical features:\n"

    cats_to_add = []
    for key, value in cats.items():
        cats_to_add.append(f"  {key}:")
        cats_to_add.append("  - " + " | ".join(value))
        # for val in value:
        #     cats_to_add.append(f"  - {val}")

    cats_to_add.append("  lesion_1:")
    cats_to_add.append("  - lesion types (see /features_lesion)")

    CATS_STR += "\n".join(cats_to_add)
    return "\n\n".join([NUMS_STR, YES_NO_STR, CATS_STR])


def model_features_lesion(model: Model) -> str:
    """Отдельное описание возможных значений параметра lesion_1;

    Вынесено в отдельную комманду, для облегчения представления остальной части параметров;

    Args:
        model (Model): текущая модель, поддерживает Model.get_features_dict();

    Returns:
        str: строка-сообщение с перечислением возможных значений lesion_1.
    """
    d = model.get_features_dict()
    return f"lesion_1 possible values:\n{', '.join(d['lesion_1'])}"
