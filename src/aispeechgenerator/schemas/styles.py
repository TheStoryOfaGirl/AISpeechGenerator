from pydantic import BaseModel


class SpeechStyle(BaseModel):
    """
    Модель стиля выступления с названием и описанием.

    Используется для определения доступных стилей речи и их характеристик.
    Стили влияют на тон, структуру и лексику генерируемой речи.

    Attributes:
        name: Уникальное название стиля (например, "formal", "inspirational", "conversational").
        description: Подробное описание стиля, объясняющее его особенности и использование.

    Examples:
        >>> formal_style = SpeechStyle(
        ...     name="formal",
        ...     description="Официальный стиль с строгой структурой, профессиональной лексикой..."
        ... )
        >>> formal_style.name
        'formal'
        >>> len(formal_style.description) > 0
        True

    Note:
        Названия стилей должны быть уникальными в рамках коллекции стилей.
        Описания должны быть достаточно подробными, чтобы модель могла понять различия между стилями.
    """

    name: str
    description: str
