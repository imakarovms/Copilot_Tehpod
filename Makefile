.PHONY: help install download-models index test test-security eval diagnose api interactive clean format lint

# ============================================================================
# Установка и настройка
# ============================================================================

install: ## Создать виртуальное окружение и установить зависимости
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✓ Окружение установлено"

download-models: ## Скачать модели (MiniLM ~80MB + Qwen2.5-7B ~4.7GB)
	@echo "Скачивание моделей..."
	. venv/bin/activate && python scripts/download_model.py
	. venv/bin/activate && python scripts/download_llm.py
	@echo "✓ Модели скачаны"

index: ## Построить индекс тикетов в ChromaDB
	. venv/bin/activate && python scripts/build_index.py
	@echo "✓ Индекс построен"

setup: install download-models index ## Полная настройка проекта (установка + модели + индекс)
	@echo "✓ Проект полностью настроен"


# Запуск
interactive: ## Запустить интерактивный режим (модели загружаются один раз)
	. venv/bin/activate && python -m cli.interactive

api: ## Запустить FastAPI сервер (Swagger UI: http://localhost:8000/docs)
	. venv/bin/activate && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

diagnose: ## Одноразовая диагностика: make diagnose ARGS="текст проблемы"
	. venv/bin/activate && python -m cli.diagnose "$(ARGS)"

# Тестирование
test: ## Запустить все тесты
	. venv/bin/activate && pytest tests/ -v

test-security: ## Запустить только тесты безопасности
	. venv/bin/activate && pytest tests/test_security.py -v

test-fast: ## Быстрые тесты (без ML-моделей)
	. venv/bin/activate && pytest tests/test_security.py tests/test_embedder.py -v -k "not test_shape and not test_semantic and not test_top_k and not test_determinism"

eval: ## Оценить качество поиска (Hit Rate)
	. venv/bin/activate && python eval/run_eval.py
	. venv/bin/activate && python eval/run_eval_rerank.py

eval-gen: ## Оценить качество генерации ответов (LLM-as-a-Judge)
	. venv/bin/activate && python eval/run_eval_generation.py
	
clean-cache: ## Очистить кеш ответов
	rm -rf cache/
	@echo "✓ Кеш очищен"