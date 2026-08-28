.PHONY: install download-models index test eval diagnose api clean

install:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

download-models:
	. venv/bin/activate && python scripts/download_model.py
	. venv/bin/activate && python scripts/download_llm.py

index:
	. venv/bin/activate && python scripts/build_index.py

test:
	. venv/bin/activate && pytest tests/ -v

eval:
	. venv/bin/activate && python eval/run_eval.py
	. venv/bin/activate && python eval/run_eval_rerank.py

diagnose:
	. venv/bin/activate && python -m cli.diagnose "$(ARGS)"

api:
	. venv/bin/activate && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf venv/ chroma_db/ models/ __pycache__/ .pytest_cache/ eval/results/
	find . -type d -name "__pycache__" -exec rm -r {} +
    
api:
	. venv/bin/activate && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000