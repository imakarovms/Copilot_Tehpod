# scripts/download_reranker.py
from sentence_transformers import CrossEncoder
from pathlib import Path

model_path = Path("models/reranker")
model_path.mkdir(parents=True, exist_ok=True)

print(f"Скачиваю модель реранкера в {model_path.resolve()} ...")
model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
model.save(str(model_path))
print("Готово! Модель сохранена локально.")