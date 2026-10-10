"""
Модуль API-роутов для генерации речей и управления параметрами модели.

Этот модуль предоставляет REST API эндпоинты для взаимодействия с генератором речей:
- генерация текста речей на основе запросов
- настройка параметров языковой модели
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from ..ai import model_parameters
from ..ai.speech_generator import SpeechGenerator
from ..dependencies import get_speech_generator
from ..schemas.model import SpeechRequest, SpeechResponse, ModelSettings
from ..utils import load_styles

# Роутер для эндпоинтов генерации речи
router = APIRouter()


@router.post("/generate_speech", response_model=SpeechResponse)
async def generate_speech(
    request: SpeechRequest,
    speech_generator: Annotated[SpeechGenerator, Depends(get_speech_generator)],
) -> SpeechResponse:
    """
    Генерирует текст речи на основе переданных параметров запроса.

    Этот эндпоинт принимает тему, стиль, длительность и другие параметры речи,
    и возвращает сгенерированный текст готовый для произнесения.

    Args:
        request (SpeechRequest): Объект запроса с параметрами речи, включая:
            - topic: Тема речи
            - duration_minutes: Длительность в минутах
            - style: Стиль выступления
            - language: Язык речи
            - key_points: Список ключевых моментов (опционально)
            - custom_instructions: Дополнительные инструкции (опционально)
        speech_generator (SpeechGenerator): Инстанс генератора речей,
            внедряемый через dependency injection.

    Returns:
        SpeechResponse: Объект ответа, содержащий сгенерированный текст речи.

    Raises:
        HTTPException: Возможные ошибки:
            - 400: Некорректный запрос
            - 422: Ошибка валидации параметров
            - 500: Ошибка генерации модели
    """

    print("Начало генерации речи")
    return SpeechResponse(
        speech=speech_generator.generate_speech(request, load_styles())
    )


@router.post("/set_model_settings")
async def set_model_settings(settings: ModelSettings) -> None:
    """
    Обновляет глобальные параметры языковой модели для генерации текста.

    Позволяет динамически менять параметры генерации без перезагрузки приложения.
    Изменения применяются ко всем последующим запросам генерации.

    Args:
        settings (ModelSettings): Объект с новыми значениями параметров:
            - do_sample (bool): Использовать ли стохастическую выборку
            - max_length (int): Максимальная длина контекста
            - max_new_tokens (int): Максимальное количество новых токенов
            - repetition_penalty (float): Штраф за повторения
            - temperature (float): Температура для выборки (креативность)
            - top_k (int): Параметр top-k выборки
            - top_p (float): Параметр top-p (nucleus) выборки

    Returns:
        None: Функция не возвращает значение, только обновляет глобальные параметры.

    Raises:
        HTTPException:
            - 422: Ошибка валидации параметров
            - 400: Некорректные значения параметров
    """
    # Обновляем параметры модели
    model_parameters.do_sample = settings.do_sample
    model_parameters.max_length = settings.max_length
    model_parameters.max_new_tokens = settings.max_new_tokens
    model_parameters.repetition_penalty = settings.repetition_penalty
    model_parameters.temperature = settings.temperature
    model_parameters.top_k = settings.top_k
    model_parameters.top_p = settings.top_p
