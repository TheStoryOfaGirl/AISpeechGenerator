"""
Модуль для генерации текстов речей с использованием языковой модели Phi-3.
Включает класс SpeechGenerator для работы с моделью и генерации речей на основе запросов.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import model_parameters
from ..schemas.model import SpeechRequest


class SpeechGenerator:
    """
    Класс для генерации речей с использованием модели Phi-3 mini.

    Этот класс обеспечивает загрузку модели, форматирование промптов и генерацию
    текста речей на основе входных параметров.

    Attributes:
        SYSTEM_PROMPT (str): Системный промпт, определяющий роль модели.
        USER_PROMPT (str): Базовый пользовательский промпт для генерации речи.
        model (AutoModelForCausalLM): Загруженная языковая модель.
        tokenizer (AutoTokenizer): Токенизатор для обработки текста.
        device (str): Устройство для вычислений ('cuda' или 'cpu').
        model_loaded (bool): Флаг загрузки модели.
    """

    SYSTEM_PROMPT = """Ты - профессиональный спичрайтер и оратор.
    Твоя задача - написать качественную, структурированную речь на заданную тему.
    Речь должна быть естественной, убедительной и подходящей для устного выступления."""
    USER_PROMPT = """Пожалуйста, напиши полноценную речь с вступлением, основной частью и заключением.
    Речь должна быть готова для непосредственного произнесения."""

    def __init__(self):
        """
        Инициализирует генератор речей.

        Определяет устройство для вычислений и устанавливает флаг загрузки модели в False.
        """

        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False

    def load_model(self):
        """
        Загружает модель Phi-3 mini и токенизатор с Hugging Face.

        Загружает предобученную модель и токенизатор, настраивает pad_token
        и определяет конфигурацию модели для генерации.

        Raises:
            Exception: Если произошла ошибка при загрузке модели.
        """

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=False,
                attn_implementation="eager",
            )
            self.model_loaded = True

        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise

    def generate_prompt(
        self, request: SpeechRequest, available_styles: dict[str, str]
    ) -> str:
        """
        Генерирует форматированный промпт для модели на основе запроса.

        Args:
            request (SpeechRequest): Объект запроса с параметрами речи.
            available_styles (Dict[str, str]): Словарь доступных стилей выступления.

        Returns:
            str: Отформатированный промпт в виде строки.

        Raises:
            ValueError: Если запрашиваемый стиль не найден в available_styles.
        """

        if request.style not in available_styles:
            raise ValueError(
                f"Стиль '{request.style}' не найден. Доступные стили: {', '.join(available_styles.keys())}"
            )
        style_description = available_styles[request.style]

        user_message = f"""
Тема речи: {request.topic}
Длительность: {request.duration_minutes} минут
Стиль выступления: {style_description}
Язык: {request.language}

"""
        if request.key_points:
            points_text = "\n".join([f"- {point}" for point in request.key_points])
            user_message += f"Ключевые моменты для раскрытия:\n{points_text}\n\n"
        if request.custom_instructions:
            user_message += (
                f"Дополнительные требования:\n{request.custom_instructions}\n\n"
            )
        user_message += self.USER_PROMPT
        chat_format = f"<|system|>\n{self.SYSTEM_PROMPT}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
        return chat_format

    def generate_speech(
        self, request: SpeechRequest, available_styles: dict[str, str]
    ) -> str:
        """
        Генерирует речь на основе запроса с использованием загруженной модели.

        Args:
            request (SpeechRequest): Объект запроса с параметрами речи.
            available_styles (Dict[str, str]): Словарь доступных стилей выступления.

        Returns:
            str: Сгенерированный текст речи.

        Raises:
            RuntimeError: Если модель не была загружена перед вызовом.
            Exception: Если произошла ошибка при генерации текста.
        """

        if not self.model_loaded:
            raise RuntimeError("Модель не загружена. Подождите.")

        prompt = self.generate_prompt(request, available_styles)

        try:
            print("Начало конфигурации")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model_parameters.max_length,  # Учитываем ограничения контекста Phi-3 mini
            ).to(self.device)

            print("Сконфигурировал tokenizer")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=model_parameters.max_new_tokens,  # Максимальная длина ответа
                    temperature=model_parameters.temperature,
                    do_sample=model_parameters.do_sample,
                    top_p=model_parameters.top_p,
                    top_k=model_parameters.top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=model_parameters.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            print("Получил ответ от модели")

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("Десериализация ответа")

            if "<|assistant|>\n" in generated_text:
                response = generated_text.split("<|assistant|>\n")[-1]
                response = response.replace("<|end|>", "").strip()
            else:
                response = generated_text[len(prompt) :].strip()

            return response

        except Exception as e:
            print(f"Ошибка при генерации речи: {e}")
            raise
