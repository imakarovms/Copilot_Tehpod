# config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # --- Эмбеддинги ---
    embedding_model: str = "models/minilm"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    embedding_max_seq_length: int = 128

    # --- Реранкинг ---
    reranker_model: str = "models/reranker"  # <-- Изменено на локальный путь

    # --- Пути ---
    data_dir: str = "data"
    chroma_dir: str = "chroma_db"
    collection_name: str = "tickets"

    # --- Секреты ---
    anthropic_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


# ЭТА СТРОКА ОБЯЗАТЕЛЬНА: создает экземпляр, который читает .env
settings = Settings()
