from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from aispeechgenerator.main import app

client = TestClient(app)


class TestGenerateSpeechEndpoint:
    """Тесты для endpoint генерации речи"""

    def test_generate_speech_success(
        self, sample_speech_request, mock_speech_generator, mock_load_styles
    ):
        """Тест успешной генерации речи"""

        response = client.post(
            "/api/model/generate_speech", json=sample_speech_request.model_dump()
        )

        assert response.status_code == 200

        response_data = response.json()
        assert "speech" in response_data
        assert response_data["speech"] == "Это сгенерированная тестовая речь."

    def test_generate_speech_model_not_loaded(self, sample_speech_request):
        """Тест ошибки при незагруженной модели"""
        mock_instance = Mock()
        mock_instance.model_loaded = False
        mock_instance.generate_speech.side_effect = RuntimeError(
            "Модель не загружена. Подождите."
        )
        with pytest.raises(RuntimeError):
            with patch(
                "aispeechgenerator.dependencies._speech_generator", mock_instance
            ):
                response = client.post(
                    "/api/model/generate_speech",
                    json=sample_speech_request.model_dump(),
                )
                assert response.status_code == 500

    def test_generate_speech_missing_required_field(self):
        """Тест ошибки при отсутствии обязательного поля"""

        with patch("aispeechgenerator.dependencies._speech_generator"):
            response = client.post(
                "/api/model/generate_speech",
                json={"duration_minutes": 5, "style": "formal", "language": "ru"},
            )

            assert response.status_code == 422
