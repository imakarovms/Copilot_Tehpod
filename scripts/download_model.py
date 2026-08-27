# scripts/download_model.py
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Путь, куда сохраним модель (относительно корня проекта)
model_path = Path("models/minilm")
model_path.mkdir(parents=True, exist_ok=True)

print(f"Скачиваю модель в {model_path.resolve()} ...")
# Скачиваем и сразу сохраняем в указанную папку
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model.save(str(model_path))
print("Готово! Модель сохранена локально.")