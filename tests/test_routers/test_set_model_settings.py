import pytest
from fastapi.testclient import TestClient

from aispeechgenerator.ai import model_parameters
from aispeechgenerator.main import app

client = TestClient(app)


class TestModelParameters:
    def test_all_parameters_exist(self, model_parameters):
        """Тест что все параметры существуют и имеют значения"""
        expected_attrs = [
            "do_sample",
            "max_length",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
        ]

        for attr in expected_attrs:
            assert hasattr(model_parameters, attr)
            assert getattr(model_parameters, attr) is not None

    def test_parameters_are_importable(self):
        """Тест что все параметры можно импортировать напрямую"""
        # from ai.model_parameters import (
        #     do_sample, max_length, max_new_tokens, temperature,
        #     top_p, top_k, repetition_penalty
        # )

        assert model_parameters.do_sample is True
        assert model_parameters.max_length == 2048
        assert model_parameters.max_new_tokens == 2048
        assert model_parameters.temperature == 0.7
        assert model_parameters.top_p == 0.9
        assert model_parameters.top_k == 50
        assert model_parameters.repetition_penalty == 1.1

    def test_module_direct_access(self, model_parameters_module):
        """Тест прямого доступа к параметрам через модуль"""
        assert model_parameters_module.do_sample is True
        assert model_parameters_module.max_length == 2048
        assert model_parameters_module.max_new_tokens == 2048
        assert model_parameters_module.temperature == 0.7
        assert model_parameters_module.top_p == 0.9
        assert model_parameters_module.top_k == 50
        assert model_parameters_module.repetition_penalty == 1.1


class TestSetModelSettings:
    def test_set_all_parameters(self, sample_model_parameters):
        """Тест установки всех параметров модели"""

        new_model_parameters = sample_model_parameters.model_dump()
        response = client.post(
            "/api/model/set_model_settings", json=new_model_parameters
        )

        assert response.status_code == 200
        assert response.json() is None

        assert model_parameters.do_sample == new_model_parameters["do_sample"]
        assert model_parameters.max_length == new_model_parameters["max_length"]
        assert model_parameters.max_new_tokens == new_model_parameters["max_new_tokens"]
        assert model_parameters.temperature == new_model_parameters["temperature"]
        assert model_parameters.top_p == new_model_parameters["top_p"]
        assert model_parameters.top_k == new_model_parameters["top_k"]
        assert (
            model_parameters.repetition_penalty
            == new_model_parameters["repetition_penalty"]
        )

    @pytest.fixture(autouse=True)
    def reset_parameters(self):
        """Фикстура для сброса параметров к значениям по умолчанию после каждого теста"""
        original_settings = {
            "do_sample": True,
            "max_length": 2048,
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }

        yield

        model_parameters.do_sample = original_settings["do_sample"]
        model_parameters.max_length = original_settings["max_length"]
        model_parameters.max_new_tokens = original_settings["max_new_tokens"]
        model_parameters.temperature = original_settings["temperature"]
        model_parameters.top_p = original_settings["top_p"]
        model_parameters.top_k = original_settings["top_k"]
        model_parameters.repetition_penalty = original_settings["repetition_penalty"]

    def test_endpoint_exists(self):
        """Тест что endpoint /api/model/set_model_settings существует"""
        response = client.post("/api/model/set_model_settings", json={})
        assert response.status_code != 404
