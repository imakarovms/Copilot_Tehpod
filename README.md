# Incident Copilot

RAG-система для автоматической диагностики технических инцидентов на основе исторических тикетов.

## Что делает система

Принимает описание проблемы от пользователя, находит релевантные исторические тикеты в базе знаний и генерирует структурированный ответ с указанием причины и решения, ссылаясь на конкретные тикеты.

**Пример работы:**

Пользователь: "Не подключается VPN, выдает ошибку таймаута"

Система:  
Диагноз: Истёк клиентский сертификат VPN.  
Решение: Перевыпустить сертификат, переустановить профиль VPN-клиента.  
Источники: [T001], [T002]

## Метрики качества

| Этап | Метод | Hit Rate @ 3 |
|------|-------|--------------|
| Baseline | Семантический поиск (MiniLM) | 50.00% |
| + BM25 | Гибридный поиск (RRF) | 75.00% |
| + Лемматизация | pymorphy3 для русского языка | 87.50% |
| **Финал** | **Гибридный поиск + RRF + Лемматизация** | **100.00%** |

Улучшение: +50 процентных пунктов за счёт комбинации методов.

## Архитектура
+---------------------------------------------------------------+
|                      IncidentPipeline                         |
+---------------------------------------------------------------+
|                                                               |
|  1. SecurityValidator ---> Валидация запроса                  |
|         |                                                     |
|         v                                                     |
|  2. Retriever ------------> Гибридный поиск                   |
|     |- Semantic (MiniLM) ---> Векторный поиск в ChromaDB      |
|     |- BM25 (pymorphy3) ---> Лексический поиск с леммами      |
|         |                      |                              |
|         +------ RRF -----------+                              |
|                    |                                          |
|                    v                                          |
|  3. Reranker -----------> Эвристический буст категорий        |
|         |                                                     |
|         v                                                     |
|  4. Generator ------------> Локальная LLM (Qwen2.5-7B)        |
|         |                                                     |
|         v                                                     |
|  5. SecurityValidator ---> Валидация ответа                   |
|                                                               |
+---------------------------------------------------------------+


## Быстрый старт

### Требования
- Python 3.12+
- CUDA-совместимая видеокарта (рекомендуется RTX 4060 8GB+)
- ~6 ГБ свободного места для моделей

### Установка
```bash
git clone https://github.com/ваш-username/incident-copilot.git
cd incident-copilot

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python scripts/download_model.py
python scripts/download_llm.py

python scripts/build_index.py

API сервер:
make api

Откройте Swagger UI: http://localhost:8000/docs

CLI (диагностика из терминала):
python -m cli.diagnose "Не работает интернет"

Оценка качества:
make eval

API
POST /diagnose
Диагностика инцидента.
Request:
{
  "query": "Не подключается VPN, выдает ошибку таймаута",
  "top_k": 3
}

Response:
{
  "answer": "Диагноз: Истёк клиентский сертификат VPN.\nРешение: Перевыпустить сертификат...",
  "citations": ["T001", "T002", "T008"],
  "confidence": "high"
}

GET /health
Проверка работоспособности.
Response:
{
  "status": "ok",
  "version": "1.0.0"
}

Структура проекта
incident-copilot/
├── api/
│   └── main.py              # FastAPI приложение
├── cli/
│   └── diagnose.py          # CLI интерфейс
├── config/
│   └── settings.py          # Конфигурация
├── data/
│   ├── index/               # Исходные тикеты
│   └── eval/                # Eval-датасет
├── eval/
│   ├── run_eval.py          # Скрипт оценки
│   └── run_eval_rerank.py   # Оценка с реранкером
── models/                  # Локальные модели (не в git)
│   ├── minilm/              # Эмбеддинги
│   └── llm/                 # Qwen2.5-7B
├── scripts/
│   ├── build_index.py       # Построение индекса
│   ├── download_model.py    # Скачивание эмбеддингов
│   └── download_llm.py      # Скачивание LLM
── src/
│   ├── embedder.py          # Эмбеддинги запросов
│   ├── generator.py         # Генерация ответов (LLM)
│   ├── indexer.py           # Индексация тикетов
│   ├── pipeline.py          # Orchestrator
│   ├── reranker.py          # Реранкинг
│   ├── retriever.py         # Гибридный поиск
│   └── security.py          # Валидация и защита
├── tests/
│   ├── test_embedder.py
│   ├── test_retriever.py
│   └── test_security.py
── .github/workflows/
│   └── lint.yml             # CI/CD
├── Makefile                 # Автоматизация
├── requirements.txt         # Зависимости
── README.md                # Этот файл

Технологии
Эмбеддинги
	paraphrase-multilingual-MiniLM-L12-v2
	    Векторизация текста (384d)
Векторная БД
	ChromaDB
	    Хранение и поиск эмбеддингов
Лексический поиск
	BM25 + pymorphy3
	    Поиск по ключевым словам с лемматизацией
Fusion
	Reciprocal Rank Fusion
	    Объединение результатов поиска
LLM
	Qwen2.5-7B-Instruct (GGUF Q4_K_M)
	    Генерация ответов
LLM Runtime
	llama-cpp-python
	    Локальный инференс на GPU
API
	FastAPI + Uvicorn
	    REST API
CI/CD
	GitHub Actions
	    Автоматический линтинг и тесты
        
Безопасность

    Валидация длины запроса (макс. 1000 символов)
    Детекция промпт-инъекций
    Обнаружение секретов (пароли, токены) в запросах
    Проверка наличия ссылок в ответах LLM