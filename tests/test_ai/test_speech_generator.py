from unittest.mock import Mock

import pytest
import torch

from aispeechgenerator.ai.speech_generator import SpeechGenerator


class TestSpeechGenerator:
    """Тесты для класса SpeechGenerator"""

    @pytest.fixture
    def speech_generator(self):
        """Фикстура для SpeechGenerator с замоканной моделью"""
        generator = SpeechGenerator()
        generator.model_loaded = True

        input_data = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_inputs = Mock()
        mock_inputs.to.return_value = input_data

        generator.tokenizer = Mock()
        generator.tokenizer.return_value = mock_inputs
        generator.tokenizer.decode.return_value = (
            "<|assistant|>\nТестовая сгенерированная речь<|end|>"
        )

        generator.model = Mock()
        generator.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        generator.device = "cpu"
        return generator

    def test_generate_prompt_success(
        self, speech_generator, sample_speech_request, sample_available_styles
    ):
        """Тест успешной генерации промпта"""

        prompt = speech_generator.generate_prompt(
            sample_speech_request, sample_available_styles
        )

        assert isinstance(prompt, str)
        assert sample_speech_request.topic in prompt
        assert sample_available_styles[sample_speech_request.style] in prompt
        assert speech_generator.USER_PROMPT in prompt

    def test_generate_prompt_invalid_style(
        self, speech_generator, sample_speech_request
    ):
        """Тест ошибки при невалидном стиле"""

        sample_speech_request.style = "invalid_style"
        available_styles = {"formal": "Формальный стиль"}

        with pytest.raises(ValueError, match="Стиль 'invalid_style' не найден"):
            speech_generator.generate_prompt(sample_speech_request, available_styles)

    def test_generate_speech_success(
        self, speech_generator, sample_speech_request, sample_available_styles
    ):
        """Тест успешной генерации речи"""

        result = speech_generator.generate_speech(
            sample_speech_request, sample_available_styles
        )

        assert result == "Тестовая сгенерированная речь"
        speech_generator.tokenizer.assert_called()
        speech_generator.model.generate.assert_called_once()

    def test_generate_speech_model_not_loaded(
        self, sample_speech_request, sample_available_styles
    ):
        """Тест ошибки при незагруженной модели"""

        generator = SpeechGenerator()
        generator.model_loaded = False

        with pytest.raises(RuntimeError, match="Модель не загружена"):
            generator.generate_speech(sample_speech_request, sample_available_styles)

    def test_generate_speech_model_generation_error(
        self, speech_generator, sample_speech_request, sample_available_styles
    ):
        """Тест ошибки при генерации модели"""

        speech_generator.model.generate.side_effect = RuntimeError("Generation error")

        with pytest.raises(RuntimeError):
            speech_generator.generate_speech(
                sample_speech_request, sample_available_styles
            )
