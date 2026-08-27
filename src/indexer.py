# индексация в ChromaDB
"""
src/indexer.py — индексация тикетов в векторную базу данных ChromaDB.

Отвечает за:
- Инициализацию персистентного клиента ChromaDB.
- Преобразование тикетов в эмбеддинги и загрузку в коллекцию.
- Идемпотентное обновление (upsert) и очистку индекса.
"""
import logging

import chromadb

from config.settings import settings
from src.embedder import embed_texts, ticket_to_text

logger = logging.getLogger(__name__)


class Indexer:
    """
    Инкапсулирует работу с векторной базой данных ChromaDB.
    Предоставляет высокоуровневый API для добавления, поиска и управления тикетами.
    """

    def __init__(self):
        logger.info("Инициализация ChromaDB (persist_directory: %s)", settings.chroma_dir)
        # PersistentClient сохраняет данные на диск, чтобы не пересобирать индекс каждый раз
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        
        # Получаем существующую коллекцию или создаем новую.
        # Используем косинусную близость (по умолчанию в ChromaDB, но явно укажем для ясности).
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            "Коллекция '%s' готова. Текущее количество записей: %d",
            self.collection.name,
            self.collection.count()
        )

    def add_tickets(self, tickets: list[dict], batch_size: int | None = None) -> int:
        """
        Добавляет (или обновляет) тикеты в коллекцию.
        Использует upsert для обеспечения идемпотентности: повторный запуск
        с теми же ID не создаст дубликаты, а обновит существующие записи.

        Args:
            tickets: список словарей с тикетами (должны содержать 'id').
            batch_size: размер батча для эмбеддинга и загрузки. 
                        Если None, берется из settings.embedding_batch_size.

        Returns:
            Количество успешно обработанных тикетов.
        """
        if not tickets:
            logger.warning("add_tickets: передан пустой список тикетов.")
            return 0

        batch_size = batch_size or settings.embedding_batch_size
        total_added = 0

        # Разбиваем на батчи, чтобы избежать OOM при генерации эмбеддингов и загрузке в БД
        for i in range(0, len(tickets), batch_size):
            batch_tickets = tickets[i:i + batch_size]
            
            # 1. Превращаем тикеты в текст для эмбеддинга
            texts = [ticket_to_text(t) for t in batch_tickets]
            
            # 2. Генерируем векторы (embed_texts уже возвращает нормализованный numpy array)
            embeddings = embed_texts(texts)
            
            # 3. Формируем ID (ChromaDB требует строковые ID)
            ids = [str(t["id"]) for t in batch_tickets]
            
            # 4. Формируем метаданные. 
            # ChromaDB не принимает None в метаданных, поэтому фильтруем и приводим к строке.
            metadatas = []
            for t in batch_tickets:
                meta = {}
                for key in ["category", "title", "description", "root_cause", "resolution"]:
                    val = t.get(key)
                    if val is not None:
                        meta[key] = str(val)
                metadatas.append(meta)

            # 5. Загружаем в ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,      # ChromaDB отлично принимает numpy arrays
                metadatas=metadatas,
                documents=texts             # Сохраняем исходный текст для удобства извлечения
            )
            
            total_added += len(batch_tickets)
            logger.debug("Загружен батч %d-%d из %d", i, i + len(batch_tickets), len(tickets))

        logger.info("Успешно обработано %d тикетов. Всего в коллекции: %d", 
                    total_added, self.collection.count())
        return total_added

    def get_count(self) -> int:
        """Возвращает текущее количество документов в коллекции."""
        return self.collection.count()

    def clear(self) -> None:
        """
        Полностью удаляет коллекцию. 
        Используйте для чистой пересборки индекса с нуля.
        """
        logger.warning("Очистка коллекции '%s'...", self.collection.name)
        self.client.delete_collection(self.collection.name)
        
        # Пересоздаем пустую коллекцию, чтобы объект self.collection оставался валидным
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Коллекция очищена и пересоздана.")

    def get_all_ids(self) -> list[str]:
        """Возвращает список всех ID в коллекции (полезно для валидации и сверки)."""
        result = self.collection.get(include=[])
        return result["ids"]