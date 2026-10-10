"""
Основной модуль FastAPI приложения для генерации речей.

Этот модуль инициализирует FastAPI приложение, настраивает роутеры и управляет
жизненным циклом приложения. Также содержит точку входа для запуска сервера.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from .dependencies import init_speech_generator
from .routers.model_api import router as model_router
from .routers.styles_api import router as style_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер жизненного цикла приложения FastAPI.

    Управляет инициализацией и завершением ресурсов приложения.
    Вызывается при старте и остановке приложения.

    Args:
        app (FastAPI): Экземпляр FastAPI приложения.

    Yields:
        None: Контроль возвращается FastAPI для работы приложения.

    Side Effects:
        - Инициализирует генератор речей при старте приложения
        - Высвобождает ресурсы при завершении (если будут добавлены)
    """
    # Инициализация при старте приложения
    init_speech_generator()
    yield
    # Здесь можно добавить код для очистки при завершении


# Создание основного экземпляра FastAPI приложения
app = FastAPI(
    title="Speech Generation API",
    description="API для генерации речей для выступлений",
    lifespan=lifespan,
)

# Подключение роутеров API с префиксами
app.include_router(model_router, prefix="/api/model")
app.include_router(style_router, prefix="/api/styles")


if __name__ == "__main__":
    """
    Точка входа для запуска сервера разработки.

    Запускает сервер Uvicorn с настройками по умолчанию:
    - Хост: 0.0.0.0 (доступ со всех интерфейсов)
    - Порт: 8000

    Используется только для разработки. В продакшене используется ASGI сервер.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)
