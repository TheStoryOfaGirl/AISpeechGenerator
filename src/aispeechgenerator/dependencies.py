"""
Модуль управления зависимостью SpeechGenerator для FastAPI приложения.

Этот модуль реализует паттерн Dependency Injection для инициализации и предоставления
единого экземпляра генератора речей во всем приложении. Используется глобальная
переменная для хранения инициализированного экземпляра SpeechGenerator.
"""

from .ai.speech_generator import SpeechGenerator

# Глобальная переменная для хранения единственного экземпляра SpeechGenerator
# Используется для реализации паттерна Singleton
_speech_generator = None


async def get_speech_generator() -> SpeechGenerator:
    """
    Dependency provider для внедрения SpeechGenerator в эндпоинты FastAPI.

    Эта функция используется как зависимость (dependency) в FastAPI маршрутах
    для предоставления доступа к инициализированному генератору речей.

    Returns:
        SpeechGenerator: Инициализированный экземпляр генератора речей.

    Raises:
        RuntimeError: Если генератор не был инициализирован перед использованием.

    Пример использования в FastAPI:
        @app.post("/generate")
        async def generate_speech(
            generator: SpeechGenerator = Depends(get_speech_generator)
        ):
            ...
    """

    if _speech_generator is None:
        raise RuntimeError("SpeechGenerator не инициализирован")
    return _speech_generator


def init_speech_generator():
    """
    Инициализирует SpeechGenerator при старте приложения.

    Эта функция должна быть вызвана при запуске приложения для загрузки
    модели и создания экземпляра SpeechGenerator. Загруженный экземпляр
    сохраняется в глобальной переменной для последующего использования.

    Процесс инициализации:
        1. Создает экземпляр SpeechGenerator
        2. Загружает модель и токенизатор
        3. Сохраняет экземпляр в глобальной переменной

    Side Effects:
        - Изменяет глобальную переменную _speech_generator
        - Выводит сообщения о процессе загрузки в консоль

    Raises:
        Exception: Если произошла ошибка при загрузке модели.
    """

    print("Начало загрузки модели...")
    global _speech_generator
    _speech_generator = SpeechGenerator()
    _speech_generator.load_model()
    print("Модель загружена")
