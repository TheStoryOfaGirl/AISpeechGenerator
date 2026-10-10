from unittest.mock import Mock, patch

import pytest

from aispeechgenerator.ai.model_parameters import (
    do_sample,
    max_length,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
)
from aispeechgenerator.schemas.model import ModelSettings
from aispeechgenerator.schemas.model import SpeechRequest


@pytest.fixture
def sample_speech_request():
    """Фикстура для тестового запроса"""
    return SpeechRequest(
        topic="Технологии будущего",
        duration_minutes=5,
        style="formal",
        language="ru",
        key_points=["Искусственный интеллект", "Робототехника", "Биотехнологии"],
        custom_instructions="Сделать акцент на этические аспекты",
    )


@pytest.fixture
def sample_available_styles():
    """Фикстура для доступных стилей"""
    return {
        "formal": "Формальный стиль выступления",
        "casual": "Неформальный стиль выступления",
        "inspirational": "Вдохновляющий стиль",
    }


@pytest.fixture
def mock_speech_generator():
    """Фикстура для мокинга SpeechGenerator"""
    mock_instance = Mock()
    mock_instance.model_loaded = True
    mock_instance.generate_speech.return_value = "Это сгенерированная тестовая речь."

    with patch("aispeechgenerator.dependencies._speech_generator", mock_instance):
        yield mock_instance


@pytest.fixture
def mock_load_styles():
    """Фикстура для мокинга функции load_styles"""
    with patch("aispeechgenerator.utils.load_styles") as mock_load:
        mock_load.return_value = {
            "formal": "Формальный стиль выступления",
            "casual": "Неформальный стиль выступления",
        }
        yield mock_load


@pytest.fixture
def model_parameters():
    """Фикстура возвращает все параметры модели по умолчанию"""
    return ModelSettings(
        do_sample=do_sample,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )


@pytest.fixture
def sample_model_parameters():
    """Фикстура возвращает тестовые параметры модели"""
    return ModelSettings(
        do_sample=False,
        max_length=1024,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.8,
        top_k=30,
        repetition_penalty=1.2,
    )


@pytest.fixture
def model_parameters_module():
    """Фикстура возвращает сам модуль с параметрами"""
    from aispeechgenerator.ai import model_parameters

    return model_parameters
