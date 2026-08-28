"""
src/security.py — валидация и защита LLM-пайплайна.
"""

import re
import logging

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 1000
MAX_CONTEXT_LENGTH = 4000

SECRET_PATTERNS = [
    re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"token\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9\-_]+"),
]

INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "system prompt",
    "dan mode",
    "игнорируй инструкции",
    "будь даном",
]


class SecurityValidator:
    @staticmethod
    def validate_query(query: str) -> str:
        """Проверяет и очищает входящий запрос."""
        if not query or not isinstance(query, str):
            raise ValueError("Запрос не может быть пустым")

        query = query.strip()

        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Запрос слишком длинный: {len(query)} символов "
                f"(максимум {MAX_QUERY_LENGTH})"
            )

        query_lower = query.lower()
        for keyword in INJECTION_KEYWORDS:
            if keyword in query_lower:
                logger.warning("Потенциальная промпт-инъекция: %s...", query[:50])

        for pattern in SECRET_PATTERNS:
            if pattern.search(query):
                logger.warning("Обнаружен потенциальный секрет в запросе")

        return query

    @staticmethod
    def validate_output(answer: str, citations: list[str]) -> dict:
        """Проверяет корректность ответа модели."""
        if not answer:
            return {"is_valid": False, "reason": "Пустой ответ"}

        if "INSUFFICIENT" not in answer.upper() and not citations:
            logger.warning("Ответ без ссылок: %s...", answer[:50])
            return {"is_valid": False, "reason": "Отсутствуют ссылки на источники"}

        for cid in citations:
            if f"[{cid}]" not in answer:
                logger.warning("Ссылка %s отсутствует в тексте ответа", cid)

        return {"is_valid": True, "reason": "OK"}
