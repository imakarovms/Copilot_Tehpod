"""
src/security.py — гибридная защита LLM-пайплайна (PII + инъекции + валидация).
"""
import logging
from security.pii_detector import PIIDetector
from security.heuristic_detector import HeuristicDetector

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 1000
RISK_THRESHOLD = 0.75


class SecurityValidator:
    def __init__(self):
        self.pii = PIIDetector()
        self.heuristic = HeuristicDetector()

    def validate_query(self, query: str) -> dict:
        """
        Полная проверка запроса: PII, инъекции, длина.
        Возвращает dict с полями safe, risk_score, label, redacted_text.
        """
        if not query or not isinstance(query, str):
            return {"safe": False, "risk_score": 1.0, "label": "empty_query", "redacted_text": ""}

        query = query.strip()

        if len(query) > MAX_QUERY_LENGTH:
            return {
                "safe": False,
                "risk_score": 1.0,
                "label": "too_long",
                "redacted_text": "",
            }

        # 1. PII-детекция
        pii_entities = self.pii.detect(query)
        redacted = self.pii.redact(query, pii_entities) if pii_entities else query

        # 2. Эвристическая проверка на инъекции
        hits = self.heuristic.scan(query)
        risk = self.heuristic.risk_score(hits)

        # 3. Бонус за PII
        if pii_entities:
            risk = max(risk, 0.60)

        safe = risk < RISK_THRESHOLD
        label = "benign"
        if not safe:
            if hits:
                label = max(hits, key=lambda h: h.score).label
            elif pii_entities:
                label = "pii_detected"

        if not safe:
            logger.warning(
                "Блокировка запроса: label=%s risk=%.2f hits=%d pii=%d",
                label, risk, len(hits), len(pii_entities),
            )

        return {
            "safe": safe,
            "risk_score": round(risk, 4),
            "label": label,
            "redacted_text": redacted,
            "pii_count": len(pii_entities),
            "hit_count": len(hits),
        }

    def validate_output(self, answer: str, citations: list[str]) -> dict:
        """Проверка ответа модели на утечки."""
        if not answer:
            return {"is_valid": False, "reason": "Пустой ответ"}

        leak_markers = [
            "system prompt", "internal instruction",
            "api_key", "secret", "token:", "BEGIN PRIVATE",
        ]
        has_leak = any(k.lower() in answer.lower() for k in leak_markers)

        if has_leak:
            return {"is_valid": False, "reason": "Обнаружена утечка системных данных"}

        if "INSUFFICIENT" not in answer.upper() and not citations:
            return {"is_valid": False, "reason": "Отсутствуют ссылки на источники"}

        for cid in citations:
            if f"[{cid}]" not in answer:
                logger.warning("Ссылка %s отсутствует в тексте ответа", cid)

        return {"is_valid": True, "reason": "OK"}