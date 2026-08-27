"""Юнит-тесты embedder. Запуск: pytest tests/test_embedder.py -v"""
import numpy as np
import pytest

from src.embedder import (
    embed_query,
    embed_texts,
    ticket_to_text,
    top_k_by_similarity,
)


TICKET = {
    "id": "T001",
    "category": "network",
    "title": "Не подключается VPN",
    "description": "Пишет ошибку таймаута.",
    "root_cause": "Сертификат истёк.",
    "resolution": "Перевыпустить сертификат.",
}


class TestTicketToText:
    def test_full_ticket(self):
        text = ticket_to_text(TICKET)
        assert text == "Не подключается VPN. Пишет ошибку таймаута."

    def test_root_cause_not_in_text(self):
        """root_cause и resolution НЕ должны попадать в текст эмбеддинга."""
        text = ticket_to_text(TICKET)
        assert "сертификат" not in text.lower()

    def test_ticket_without_title(self):
        """Eval-тикеты без title тоже обрабатываются."""
        t = {"id": "E001", "description": "Просто описание."}
        assert ticket_to_text(t) == "Просто описание."


class TestEmbedTexts:
    def test_shape_and_norm(self):
        emb = embed_texts(["Первый текст", "Второй текст"])
        assert emb.shape == (2, 384)
        # все векторы нормализованы
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            embed_texts([])

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            embed_texts(["нормальный", 123])


class TestSimilarity:
    def test_semantic_close_beats_far(self):
        """Похожий текст должен быть ближе непохожего — ядро всей идеи."""
        query = embed_query("VPN не подключается, таймаут")
        base = embed_texts([
            "Не могу установить VPN-соединение, ошибка",   # похожий
            "Принтер печатает пустые листы",               # другой домен
        ])
        sim = base @ query
        assert sim[0] > sim[1]

    def test_top_k_order(self):
        query = embed_query("Проблема с оплатой, двойное списание")
        base = embed_texts([
            "Принтер не печатает",                # 0: нерелевантный
            "Двойное списание за подписку",       # 1: релевантный
            "Wi-Fi отваливается",                 # 2: нерелевантный
        ])
        top = top_k_by_similarity(query, base, k=2)
        assert top[0][0] == 1           # лучший — релевантный индекс
        assert top[0][1] >= top[1][1]   # скоры по убыванию

    def test_determinism(self):
        """Один текст → одинаковый вектор (детерминизм inference)."""
        a = embed_texts(["Проверка детерминизма"])[0]
        b = embed_texts(["Проверка детерминизма"])[0]
        assert np.allclose(a, b)