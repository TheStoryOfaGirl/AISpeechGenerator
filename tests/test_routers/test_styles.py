import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aispeechgenerator import utils
from aispeechgenerator.main import app

possible_files = [
    Path.cwd() / "speech_styles.json",
    Path(__file__).parent / "speech_styles.json",
]

for file_path in possible_files:
    if file_path.exists():
        file_path.unlink()

TEST_STYLES_FILE = Path(tempfile.gettempdir()) / "speech_styles_test.json"


def load_styles() -> dict:
    if not TEST_STYLES_FILE.exists():
        return {}
    try:
        with open(TEST_STYLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_styles(styles: dict):
    with open(TEST_STYLES_FILE, "w", encoding="utf-8") as f:
        json.dump(styles, f, indent=4, ensure_ascii=False)


utils.load_styles = load_styles
utils.save_styles = save_styles

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_styles():
    """Сбрасываем ВСЕ возможные файлы со стилями перед каждым тестом"""
    if TEST_STYLES_FILE.exists():
        TEST_STYLES_FILE.unlink()

    for file_path in possible_files:
        if file_path.exists():
            file_path.unlink()

    yield

    if TEST_STYLES_FILE.exists():
        TEST_STYLES_FILE.unlink()


class TestStylesAPI:
    """Тесты для api стилей"""

    def test_initial_get_returns_empty(self):
        """Проверяет, что изначально API возвращает пустой список стилей."""
        response = client.get("/api/styles")
        assert response.status_code == 200
        assert response.json() == {"styles": {}}

    def test_post_new_styles_success(self):
        """Проверяет успешное добавление новых стилей через POST-запрос."""
        payload = [
            {"name": "professional", "description": "Официальный деловой стиль"},
            {"name": "motivational", "description": "Вдохновляющий и энергичный"},
            {"name": "friendly", "description": "Дружелюбный, неформальный"},
        ]
        response = client.post("/api/styles", json=payload)
        assert response.status_code == 200

    def test_get_after_post_returns_correct_data(self):
        """Проверяет, что GET после POST возвращает корректные добавленные данные."""
        payload = [
            {"name": "professional", "description": "Официальный деловой стиль"},
            {"name": "motivational", "description": "Вдохновляющий и энергичный"},
            {"name": "friendly", "description": "Дружелюбный, неформальный"},
        ]
        client.post("/api/styles", json=payload)

        response = client.get("/api/styles")
        assert response.status_code == 200
        styles = response.json()["styles"]
        assert styles["professional"] == "Официальный деловой стиль"
        assert styles["motivational"] == "Вдохновляющий и энергичный"
        assert styles["friendly"] == "Дружелюбный, неформальный"

    def test_post_duplicate_style_fails(self):
        """Проверяет, что добавление дубликата стиля возвращает ошибку 400."""
        payload = [{"name": "professional", "description": "Официальный деловой стиль"}]
        client.post("/api/styles", json=payload)

        response = client.post(
            "/api/styles", json=[{"name": "professional", "description": "дубликат"}]
        )
        assert response.status_code == 400

    def test_put_updates_existing_style(self):
        """Проверяет успешное обновление существующего стиля через PUT-запрос."""
        payload = [
            {"name": "motivational", "description": "Вдохновляющий и энергичный"}
        ]
        client.post("/api/styles", json=payload)

        response = client.put(
            "/api/styles", json={"name": "motivational", "description": "МОТИВАЦИЯ!!!"}
        )
        assert response.status_code == 200

    def test_put_nonexistent_style_returns_404(self):
        """Проверяет, что обновление несуществующего стиля возвращает ошибку 404."""
        response = client.put(
            "/api/styles", json={"name": "ghost", "description": "404"}
        )
        assert response.status_code == 404

    def test_get_after_update_has_new_description(self):
        """Проверяет, что после обновления стиля GET возвращает обновленное описание."""
        payload = [
            {"name": "motivational", "description": "Вдохновляющий и энергичный"}
        ]
        client.post("/api/styles", json=payload)

        client.put(
            "/api/styles", json={"name": "motivational", "description": "МОТИВАЦИЯ!!!"}
        )

        response = client.get("/api/styles")
        assert response.status_code == 200
        assert response.json()["styles"]["motivational"] == "МОТИВАЦИЯ!!!"

    def test_post_invalid_payload_missing_name(self):
        """Проверяет, что POST с невалидным payload (без name) возвращает ошибку 422."""
        response = client.post("/api/styles", json=[{"description": "без имени"}])
        assert response.status_code == 422

    def test_put_invalid_payload_missing_name(self):
        """Проверяет, что PUT с невалидным payload (без name) возвращает ошибку 422."""
        response = client.put("/api/styles", json={"description": "только описание"})
        assert response.status_code == 422
