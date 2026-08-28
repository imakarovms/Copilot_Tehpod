# генерация диагноза через локальную Qwen
"""
src/generator.py — генерация ответа с помощью локальной LLM (Qwen2.5-7B).
"""
import logging
from pathlib import Path
from llama_cpp import Llama
from src.security import SecurityValidator
from config.settings import settings

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self):
        model_path = Path("models/llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель LLM не найдена по пути {model_path}. "
                "Запустите scripts/download_llm.py"
            )

        logger.info("Загрузка локальной LLM: %s ...", model_path.name)
        logger.info(
            "Инициализация может занять 10-20 секунд (загрузка весов в VRAM)..."
        )

        # n_gpu_layers=-1 означает, что вся модель загружается в видеокарту (RTX 4060)
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,  # Полная загрузка в GPU
            n_ctx=4096,  # Размер контекста
            verbose=False,  # Отключаем спам в консоль от llama.cpp
            n_threads=4,  # Потоки CPU для препроцессинга
        )
        logger.info("Локальная LLM успешно загружена в VRAM.")

    def generate(self, query: str, retrieved_tickets: list[dict]) -> dict:
        """
        Генерирует ответ на основе запроса и найденных тикетов.
        """
        # 1. Валидация ввода
        safe_query = SecurityValidator.validate_query(query)

        if not retrieved_tickets:
            return {
                "answer": (
                    "INSUFFICIENT DATA: В базе знаний не найдено релевантных тикетов."
                ),
                "citations": [],
                "confidence": "low",
            }
        if not retrieved_tickets:
            return {
                "answer": "INSUFFICIENT DATA: В базе знаний не найдено релевантных тикетов для решения этой проблемы.",
                "citations": [],
                "confidence": "low",
            }

        # Формируем контекст из топ-3 тикетов
        context_parts = []
        citations = []
        for ticket in retrieved_tickets[:3]:
            tid = ticket.get("id", "UNKNOWN")
            title = ticket.get("title", "Без заголовка")
            desc = ticket.get("description", "")
            resolution = ticket.get("resolution", "Решение не указано")

            context_parts.append(
                f"[{tid}] {title}\nОписание: {desc}\nРешение: {resolution}"
            )
            citations.append(tid)

        context_text = "\n\n".join(context_parts)

        # Системный промпт для Qwen (формат ChatML)
        system_prompt = (
            "Ты опытный инженер технической поддержки 2-й линии. "
            "Твоя задача — диагностировать проблему пользователя на основе предоставленных исторических тикетов. "
            "Правила:\n"
            "1. Отвечай кратко, по делу, на русском языке.\n"
            "2. Обязательно ссылайся на ID тикетов в квадратных скобках, например: [T001].\n"
            "3. Если предоставленные тикеты не помогают решить проблему, честно ответь: 'INSUFFICIENT DATA'.\n"
            "4. Не выдумывай факты, которых нет в контексте."
        )

        user_prompt = (
            f"Проблема пользователя: {query}\n\n"
            f"Релевантные исторические тикеты:\n{context_text}"
        )

        # Формируем сообщения в формате ChatML, который понимает Qwen
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("Генерация ответа для запроса: '%s...'", query[:40])

        # Генерация
        output = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.1,  # Низкая температура для фактологичности (RAG)
            max_tokens=512,  # Ограничиваем длину ответа
            stop=["<|im_end|>"],  # Стоп-токен для Qwen
        )

        answer_text = output["choices"][0]["message"]["content"].strip()
        citations = [t.get("id") for t in retrieved_tickets[:3]]

        # 3. Валидация вывода
        validation = SecurityValidator.validate_output(answer_text, citations)
        if not validation["is_valid"]:
            answer_text = (
                f"ОШИБКА ВАЛИДАЦИИ: {validation['reason']}. "
                "Пожалуйста, переформулируйте запрос."
            )
            citations = []

        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": (
                "high" if "INSUFFICIENT" not in answer_text.upper() else "low"
            ),
        }
