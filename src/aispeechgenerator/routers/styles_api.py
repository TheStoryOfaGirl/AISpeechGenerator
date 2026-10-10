"""
Модуль управления стилями выступлений.

Предоставляет API для работы со стилями речей: добавление, получение и обновление
стилей, которые могут использоваться при генерации речей.
"""

from fastapi import APIRouter, HTTPException

from ..schemas.styles import SpeechStyle
from ..utils import load_styles, save_styles

router = APIRouter()


@router.post("")
async def set_styles(styles_list: list[SpeechStyle]):
    """
    Добавляет новые стили выступлений в систему.

    Принимает список стилей и добавляет их в хранилище. Каждый стиль должен
    иметь уникальное имя. Если стиль с таким именем уже существует,
    возвращается ошибка.

    Args:
        styles_list (List[SpeechStyle]): Список объектов стилей для добавления.
            Каждый стиль содержит:
            - name (str): Уникальное имя стиля
            - description (str): Описание стиля

    Returns:
        dict: Сообщение об успешном добавлении и данные добавленных стилей.
            Пример:
            {
                "message": "Стили добавлены",
                "styles": [
                    {"name": "научный", "description": "Научный стиль речи"},
                    {"name": "художественный", "description": "Художественный стиль"}
                ]
            }

    Raises:
        HTTPException 400: Если стиль с таким именем уже существует.

    Example:
        Запрос:
        POST /styles
        [
            {
                "name": "научный",
                "description": "Научный стиль речи"
            },
            {
                "name": "художественный",
                "description": "Художественный стиль"
            }
        ]

        Ответ:
        {
            "message": "Стили добавлены",
            "styles": [
                {"name": "научный", "description": "Научный стиль речи"},
                {"name": "художественный", "description": "Художественный стиль"}
            ]
        }
    """
    styles = load_styles()
    added_styles = []
    for style in styles_list:
        if style.name in styles:
            raise HTTPException(
                status_code=400, detail=f"Стиль с именем '{style.name}' уже существует"
            )
        styles[style.name] = style.description
        added_styles.append(style.dict())
    save_styles(styles)
    return {"message": "Стили добавлены", "styles": added_styles}


@router.get("")
async def get_styles():
    """
    Получает все доступные стили выступлений из системы.

    Возвращает словарь со всеми стилями, где ключ - имя стиля,
    значение - описание стиля.

    Returns:
        dict: Словарь со всеми стилями в формате:
            {
                "имя_стиля_1": "описание_стиля_1",
                "имя_стиля_2": "описание_стиля_2",
                ...
            }

    Example:
        Запрос:
        GET /styles

        Ответ:
        {
            "научный": "Научный стиль речи",
            "художественный": "Художественный стиль",
            "официальный": "Официально-деловой стиль"
        }
    """

    styles = load_styles()
    return {"styles": styles}


@router.put("")
async def update_styles(style: SpeechStyle):
    """
    Обновляет описание существующего стиля выступления.

    Изменяет описание стиля по его имени. Если стиль с указанным именем
    не существует, возвращается ошибка.

    Args:
        style (SpeechStyle): Объект стиля для обновления, содержащий:
            - name (str): Имя существующего стиля для обновления
            - description (str): Новое описание стиля

    Returns:
        dict: Сообщение об успешном обновлении и данные обновленного стиля.
            Пример:
            {
                "message": "Стиль обновлен",
                "style": {"name": "научный", "description": "Обновленное описание..."}
            }

    Raises:
        HTTPException 404: Если стиль с указанным именем не найден.

    Example:
        Запрос:
        PUT /styles
        {
            "name": "научный",
            "description": "Обновленное описание научного стиля"
        }

        Ответ:
        {
            "message": "Стиль обновлен",
            "style": {
                "name": "научный",
                "description": "Обновленное описание научного стиля"
            }
        }
    """
    styles = load_styles()
    if style.name not in styles:
        raise HTTPException(
            status_code=404, detail=f"Стиль с именем '{style.name}' не найден"
        )
    styles[style.name] = style.description
    save_styles(styles)
    return {"message": "Стиль обновлен", "style": style.dict()}
