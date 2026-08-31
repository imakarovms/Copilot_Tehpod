# генерация диагноза через локальную Qwen
"""
src/generator.py — генерация ответа с помощью локальной LLM (Qwen2.5-7B).
"""
import logging
from pathlib import Path
from llama_cpp import Llama
from security import SecurityValidator
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
        validation = SecurityValidator.validate_query(query)

        if not validation["safe"]:
            return {
                "answer": f"Запрос отклонён: {validation['label']}. "
                        f"Уберите персональные данные или подозрительные конструкции.",
                "citations": [],
                "confidence": "low",
                "risk_score": validation["risk_score"],
            }

        safe_query = validation["redacted_text"]

        if not retrieved_tickets:
            return {
                "answer": "INSUFFICIENT DATA: В базе знаний не найдено релевантных тикетов.",
                "citations": [],
                "confidence": "low",
                "risk_score": 0.0,
            }

        # 2. Формирование контекста (без изменений)
        context_parts = []
        citations = []
        for ticket in retrieved_tickets[:3]:
            tid = ticket.get("id", "UNKNOWN")
            title = ticket.get("title", "Без заголовка")
            desc = ticket.get("description", "")
            resolution = ticket.get("resolution", "Решение не указано")
            context_parts.append(f"[{tid}] {title}\nОписание: {desc}\nРешение: {resolution}")
            citations.append(tid)

        context_text = "\n\n".join(context_parts)

        system_prompt = (
            "Ты опытный инженер технической поддержки 2-й линии. "
            "Диагностируй проблему на основе исторических тикетов. "
            "Правила:\n"
            "1. Отвечай кратко, по делу, на русском языке.\n"
            "2. Ссылайся на ID тикетов в квадратных скобках: [T001].\n"
            "3. Если тикеты не помогают — ответь 'INSUFFICIENT DATA'.\n"
            "4. Не выдумывай факты вне контекста."
        )

        user_prompt = (
            f"Проблема пользователя: {safe_query}\n\n"
            f"Релевантные исторические тикеты:\n{context_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("Генерация ответа для: '%s...'", safe_query[:40])

        output = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            stop=["<|im_end|>"],
        )

        answer_text = output["choices"][0]["message"]["content"].strip()

        # 3. Валидация вывода
        out_validation = SecurityValidator.validate_output(answer_text, citations)
        if not out_validation["is_valid"]:
            answer_text = (
                f"ОШИБКА ВАЛИДАЦИИ: {out_validation['reason']}. "
                "Пожалуйста, переформулируйте запрос."
            )
            citations = []

        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": "high" if "INSUFFICIENT" not in answer_text.upper() else "low",
            "risk_score": validation["risk_score"],
        }
