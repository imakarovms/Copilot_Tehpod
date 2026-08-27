setup:
    python3 -m venv venv && venv/bin/pip install -r requirements.txt

index:
    venv/bin/python scripts/build_index.py

api:
    venv/bin/uvicorn api.main:app --reload

cli:
    venv/bin/python -m cli.diagnose "$(QUERY)"

eval:
    venv/bin/python eval/run_eval.py