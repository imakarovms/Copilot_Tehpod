.PHONY: help install download-models index test test-security eval diagnose api interactive clean format lint

# ============================================================================
# Incident Copilot - Makefile
# ============================================================================

help: ## Показать список доступных команд
	@echo "Incident Copilot - Доступные команды:"
	@echo ""
	@echo "  Установка и настройка:"
	@echo "    install          - Создать venv и установить зависимости"
	@echo "    download-models  - Скачать модели (MiniLM + Qwen2.5-7B)"
	@echo "    index            - Построить индекс тикетов"
	@echo ""
	@echo "  Запуск:"
	@echo "    interactive      - Интерактивный режим (рекомендуется)"
	@echo "    api              - Запустить FastAPI сервер на порту 8000"
	@echo "    diagnose ARGS='...' - Одноразовая диагностика"
	@echo ""
	@echo "  Тестирование:"
	@echo "    test             - Запустить все тесты"
	@echo "    test-security    - Запустить только тесты безопасности"
	@echo "    eval             - Оценить качество поиска"
	@echo ""
	@echo "  Код:"
	@echo "    format           - Отформатировать код (black)"
	@echo "    lint             - Проверить форматирование"
	@echo ""
	@echo "  Очистка:"
	@echo "    clean            - Удалить временные файлы (кроме моделей)"
	@echo "    clean-all        - Удалить всё включая модели и БД"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

# ============================================================================
# Запуск
# ============================================================================

interactive: ## Запустить интерактивный режим (модели загружаются один раз)
	. venv/bin/activate && python -m cli.interactive

api: ## Запустить FastAPI сервер (Swagger UI: http://localhost:8000/docs)
	. venv/bin/activate && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

diagnose: ## Одноразовая диагностика: make diagnose ARGS="текст проблемы"
	. venv/bin/activate && python -m cli.diagnose "$(ARGS)"

# ============================================================================
# Тестирование
# ============================================================================

test: ## Запустить все тесты
	. venv/bin/activate && pytest tests/ -v

test-security: ## Запустить только тесты безопасности
	. venv/bin/activate && pytest tests/test_security.py -v

test-fast: ## Быстрые тесты (без ML-моделей)
	. venv/bin/activate && pytest tests/test_security.py tests/test_embedder.py -v -k "not test_shape and not test_semantic and not test_top_k and not test_determinism"

eval: ## Оценить качество поиска (Hit Rate)
	. venv/bin/activate && python eval/run_eval.py
	. venv/bin/activate && python eval/run_eval_rerank.py

# ============================================================================
# Код
# ============================================================================

format: ## Отформатировать код с помощью black
	. venv/bin/activate && black .
	@echo "✓ Код отформатирован"

lint: ## Проверить форматирование кода
	. venv/bin/activate && black --check .

# ============================================================================
# Очистка
# ============================================================================

clean: ## Удалить временные файлы (кэш, __pycache__, результаты тестов)
	rm -rf __pycache__/ .pytest_cache/ eval/results/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Временные файлы удалены"

clean-all: ## Удалить всё включая модели, БД и venv (ВНИМАНИЕ: необратимо!)
	rm -rf venv/ chroma_db/ models/ __pycache__/ .pytest_cache/ eval/results/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Все артефакты удалены (требуется повторная настройка)"