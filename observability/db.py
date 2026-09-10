import sqlite3
import os
import uuid
import time

# Путь к базе данных: она будет лежать прямо в папке observability
DB_PATH = os.path.join(os.path.dirname(__file__), "metrics.db")

def get_connection():
    """Создает подключение к БД с нужными настройками."""
    conn = sqlite3.connect(DB_PATH)
    # Включаем WAL-режим (Write-Ahead Logging)
    # Это позволяет читать данные (дашборду) и писать (трекеру) одновременно без блокировок
    conn.execute("PRAGMA journal_mode=WAL;")
    # Включаем поддержку внешних ключей (для каскадного удаления)
    conn.execute("PRAGMA foreign_keys=ON;")
    # Чтобы результаты запросов возвращались как удобные словари, а не просто кортежи
    conn.row_factory = sqlite3.Row 
    return conn

def init_db():
    """Создает таблицы в базе данных, если их еще нет."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Главная таблица для метрик (легкая, для графиков и алертов)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS calls (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            model_name TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            tokens_in INTEGER NOT NULL DEFAULT 0,
            tokens_out INTEGER NOT NULL DEFAULT 0,
            shadow_cost_usd REAL NOT NULL DEFAULT 0.0,
            success INTEGER NOT NULL DEFAULT 1,
            error_type TEXT,
            is_evaluated INTEGER NOT NULL DEFAULT 0,
            quality_score REAL
        )
    ''')
    
    # 2. Тяжелая таблица для текстов (читается только при клике на конкретный трейс)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS call_details (
            call_id TEXT PRIMARY KEY,
            input_text TEXT,
            output_text TEXT,
            judge_reasoning TEXT,
            FOREIGN KEY (call_id) REFERENCES calls(id) ON DELETE CASCADE
        )
    ''')
    
    # 3. Справочник цен для расчета "теневой стоимости"
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pricing_registry (
            model_name TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            price_in_per_1m_usd REAL NOT NULL,
            price_out_per_1m_usd REAL NOT NULL,
            effective_date TEXT NOT NULL,
            source_url TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"База данных и таблицы созданы: {DB_PATH}")

def save_call(model_name, latency_ms, tokens_in, tokens_out, success=1, error_type=None):
    """Функция для записи данных (понадобится нам в tracer.py)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    call_id = str(uuid.uuid4())
    timestamp = time.time()
    
    # Пока считаем shadow_cost = 0, логику расчета добавим на следующем шаге
    shadow_cost = 0.0 
    
    cursor.execute('''
        INSERT INTO calls (id, timestamp, model_name, latency_ms, tokens_in, tokens_out, shadow_cost_usd, success, error_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (call_id, timestamp, model_name, latency_ms, tokens_in, tokens_out, shadow_cost, success, error_type))
    
    conn.commit()
    conn.close()
    return call_id


def save_call_details(call_id: str, input_text: str, output_text: str):
    """Сохраняет тексты промпта и ответа в таблицу call_details."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO call_details (call_id, input_text, output_text)
        VALUES (?, ?, ?)
        ON CONFLICT(call_id) DO UPDATE SET 
            input_text = excluded.input_text,
            output_text = excluded.output_text
    ''', (call_id, input_text, output_text))
    
    conn.commit()
    conn.close()


# Этот блок выполняется, только если запустить файл напрямую (python db.py)
if __name__ == "__main__":
    init_db()
    
