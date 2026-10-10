from pydantic import BaseModel
from typing import List, Optional


class SpeechRequest(BaseModel):
    """
    Модель запроса для генерации речи.

    Содержит все параметры, необходимые для создания персонализированной речи
    с использованием языковой модели.

    Attributes:
        topic: Основная тема речи.
        duration_minutes: Примерная продолжительность речи в минутах.
        style: Стиль выступления. Влияет на тон и структуру речи.
               По умолчанию: "professional".
        key_points: Список ключевых моментов, которые должны быть раскрыты в речи.
                    Может быть None, если не требуется.
        language: Язык, на котором должна быть написана речь.
                  По умолчанию: "ru" (русский).
        custom_instructions: Дополнительные пожелания или требования к содержанию речи.
                             Может быть None, если не требуется.

    Examples:
        >>> request = SpeechRequest(
        ...     topic="Инновации в образовании",
        ...     duration_minutes=7,
        ...     style="inspirational",
        ...     key_points=["Цифровизация", "Персонализация", "Геймификация"],
        ...     language="ru"
        ... )
        >>> request.topic
        'Инновации в образовании'
    """

    topic: str
    duration_minutes: int
    style: str = "professional"
    key_points: Optional[List[str]] = None
    language: str = "ru"
    custom_instructions: Optional[str] = None


class SpeechResponse(BaseModel):
    """
    Модель ответа с сгенерированной речью.

    Содержит готовый текст речи, созданный языковой моделью на основе
    параметров из SpeechRequest.

    Attributes:
        speech: Текст сгенерированной речи. Включает в себя вступление,
                основную часть и заключение, отформатированные для устного выступления.

    Examples:
        >>> response = SpeechResponse(
        ...     speech="Добрый день, уважаемые коллеги! Сегодня мы обсудим..."
        ... )
        >>> len(response.speech) > 0
        True
    """

    speech: str


class ModelSettings(BaseModel):
    """
    Настройки генерации текста для языковой модели.

    Содержит гиперпараметры, которые влияют на креативность, разнообразие
    и качество сгенерированного текста.

    Attributes:
        temperature: Контролирует случайность генерации.
                     Более высокие значения (ближе к 1.0) делают вывод более случайным,
                     более низкие (ближе к 0.0) - более детерминированным.
                     По умолчанию: 0.7.
        top_p: Параметр nucleus sampling. Ограничивает выбор следующего токена
               из наиболее вероятных токенов, чья сумма вероятностей превышает p.
               По умолчанию: 0.9.
        top_k: Ограничивает выбор следующего токена k наиболее вероятными токенами.
               По умолчанию: 50.
        max_length: Максимальная общая длина входных и выходных токенов.
                    По умолчанию: 2048.
        max_new_tokens: Максимальное количество новых токенов для генерации.
                        По умолчанию: 2048.
        repetition_penalty: Штраф за повторения. Значения >1.0 уменьшают вероятность
                            повторения уже использованных токенов.
                            По умолчанию: 1.1.
        do_sample: Использовать ли случайную выборку при генерации.
                   Если False, используется жадное декодирование.
                   По умолчанию: True.

    Note:
        - Значения по умолчанию оптимизированы для модели Phi-3-mini
        - Слишком высокие значения max_length и max_new_tokens могут привести
          к превышению контекстного окна модели

    Examples:
        >>> settings = ModelSettings(temperature=0.8, top_p=0.95)
        >>> settings.temperature
        0.8
        >>> settings.do_sample
        True
    """

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_length: int = 2048
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.1
    do_sample: bool = True
