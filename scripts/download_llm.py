# scripts/download_llm.py
from huggingface_hub import hf_hub_download
from pathlib import Path

model_dir = Path("models/llm")
model_dir.mkdir(parents=True, exist_ok=True)

repo_id = "bartowski/Qwen2.5-7B-Instruct-GGUF"
filename = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

print(f"Скачиваю {filename}  в {model_dir.resolve()} ...")

file_path = hf_hub_download(
    repo_id=repo_id, filename=filename, local_dir=str(model_dir)
)

print(f"Готово! Модель сохранена: {file_path}")
