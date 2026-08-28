"""
src/retriever.py — гибридный поиск релевантных тикетов (Семантика + BM25 + RRF).
"""
import logging
import re
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi

from config.settings import settings
from src.embedder import embed_query

logger = logging.getLogger(__name__)


import re
import pymorphy3

# Инициализируем анализатор один раз при импорте модуля
morph = pymorphy3.MorphAnalyzer()

def tokenize(text: str) -> list[str]:
    """
    Токенизация с лемматизацией для русского языка.
    Превращает "с графиками" в ["с", "график"].
    """
    # Находим все слова (игнорируем пунктуацию)
    words = re.findall(r'\w+', text.lower())
    
    # Приводим каждое слово к начальной форме (нормальной форме)
    # morph.parse(word)[0] берет наиболее вероятный вариант разбора
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    
    return lemmas

class Retriever:
    def __init__(self):
        logger.info("Инициализация Retriever (подключение к %s)", settings.chroma_dir)
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection = self.client.get_collection(name=settings.collection_name)
        
        # --- Построение BM25 индекса в памяти ---
        logger.info("Построение BM25 индекса...")
        all_docs = self.collection.get(include=["documents", "metadatas"])
        
        self.corpus_ids = all_docs["ids"]
        self.corpus_metadatas = all_docs["metadatas"]
        self.corpus_documents = all_docs["documents"]
        
        # Токенизируем все документы для BM25
        tokenized_corpus = [tokenize(doc) for doc in self.corpus_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 индекс построен. Документов: %d", len(self.corpus_ids))

    def search_semantic(self, query: str, k: int = 10) -> list[dict]:
        """Семантический поиск. Возвращает до k результатов."""
        query_embedding = embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            similarity = 1.0 - results["distances"][0][i]
            ticket = {
                "id": results["ids"][0][i],
                "score": float(similarity),
                "document": results["documents"][0][i],
            }
            if results["metadatas"][0][i]:
                ticket.update(results["metadatas"][0][i])
            retrieved.append(ticket)
        return retrieved

    def search_bm25(self, query: str, k: int = 10) -> list[dict]:
        """Лексический поиск BM25. Возвращает до k результатов."""
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Сопоставляем скоры с ID и сортируем по убыванию
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        retrieved = []
        for idx, score in indexed_scores[:k]:
            if score > 0: # Игнорируем нулевые совпадения
                ticket = {
                    "id": self.corpus_ids[idx],
                    "score": float(score),
                    "document": self.corpus_documents[idx],
                }
                if self.corpus_metadatas[idx]:
                    ticket.update(self.corpus_metadatas[idx])
                retrieved.append(ticket)
        return retrieved

    def reciprocal_rank_fusion(
        self, 
        semantic_results: list[dict], 
        bm25_results: list[dict], 
        k: int = 5, 
        rrf_k: int = 60
    ) -> list[dict]:
        """
        Объединяет два списка результатов с помощью Reciprocal Rank Fusion.
        """
        rrf_scores = defaultdict(float)
        
        # Накопление скоров из семантического поиска
        for rank, item in enumerate(semantic_results):
            rrf_scores[item["id"]] += 1.0 / (rrf_k + rank + 1)
            
        # Накопление скоров из BM25
        for rank, item in enumerate(bm25_results):
            rrf_scores[item["id"]] += 1.0 / (rrf_k + rank + 1)
            
        # Сортировка по итоговому RRF скору
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Формирование финального списка
        # Создаем быстрый доступ к данным по ID
        all_results = {item["id"]: item for item in semantic_results + bm25_results}
        
        final_results = []
        for item_id in sorted_ids[:k]:
            item = all_results[item_id].copy()
            item["score"] = float(rrf_scores[item_id]) # Перезаписываем скор на RRF
            final_results.append(item)
            
        return final_results

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Основной метод гибридного поиска.
        """
        # Запрашиваем больше кандидатов у каждого метода, чтобы RRF было из чего выбирать
        candidates_k = max(k * 2, 10) 
        
        semantic_results = self.search_semantic(query, k=candidates_k)
        bm25_results = self.search_bm25(query, k=candidates_k)
        
        return self.reciprocal_rank_fusion(semantic_results, bm25_results, k=k)