"""
eval/run_eval.py — оценка качества гибридного поиска (Семантика + BM25 + RRF).
"""
import csv
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from src.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    eval_path = Path(settings.data_dir) / "eval" / "eval.json"
    if not eval_path.exists():
        logger.error("Файл eval-датасета не найден: %s", eval_path)
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    logger.info("Загружено %d запросов для оценки", len(eval_data))
    retriever = Retriever()
    
    normal_total = 0
    normal_hits = 0
    edge_total = 0
    edge_low_score_hits = 0

    results = []
    misses_for_debug = []

    for item in eval_data:
        query = item.get("description")
        expected_id = item.get("matches_index_id")
        note = item.get("note", "unknown")
        query_id = item.get("id")
        
        if not query:
            continue

        search_results = retriever.search(query, k=3)
        retrieved_ids = [res["id"] for res in search_results]
        top_score = search_results[0]["score"] if search_results else 0.0

        is_hit = False
        evaluation_note = ""

        if note == "should_match" and expected_id:
            normal_total += 1
            is_hit = expected_id in retrieved_ids
            if is_hit:
                normal_hits += 1
                evaluation_note = "OK"
            else:
                evaluation_note = f"MISS (expected {expected_id}, got {retrieved_ids})"
                misses_for_debug.append({
                    "id": query_id,
                    "query": query,
                    "expected": expected_id,
                    "got": search_results
                })
            
        elif note == "edge_case_no_match":
            edge_total += 1
            is_success = top_score < 0.65
            if is_success:
                edge_low_score_hits += 1
            evaluation_note = "OK (low score)" if is_success else f"FALSE POSITIVE (score {top_score:.2f})"

        results.append({
            "query_id": query_id,
            "query": query,
            "expected_id": expected_id or "N/A",
            "retrieved_ids": "|".join(retrieved_ids),
            "top_score": f"{top_score:.3f}",
            "note": note,
            "evaluation": evaluation_note
        })

    normal_hit_rate = (normal_hits / normal_total * 100) if normal_total > 0 else 0.0
    edge_success_rate = (edge_low_score_hits / edge_total * 100) if edge_total > 0 else 0.0

    logger.info("=" * 60)
    logger.info("HYBRID EVAL RESULTS (Semantic + BM25 + RRF)")
    logger.info("-" * 60)
    logger.info("1. Обычные запросы (should_match):")
    logger.info("   Total: %d | Hits @ 3: %d | Hit Rate: %.2f%%", normal_total, normal_hits, normal_hit_rate)
    logger.info("2. Краевые случаи (edge_case_no_match):")
    logger.info("   Total: %d | Low Score Success: %d | Success Rate: %.2f%%", edge_total, edge_low_score_hits, edge_success_rate)
    logger.info("=" * 60)

    # Диагностика первых двух промахов
    if misses_for_debug:
        logger.info("\n--- ДИАГНОСТИКА ПРОМАХОВ (первые 2) ---")
        for miss in misses_for_debug[:2]:
            logger.info("Запрос ID: %s", miss["id"])
            logger.info("Текст: %s", miss["query"][:60] + "...")
            logger.info("Ожидался: %s", miss["expected"])
            logger.info("Получено:")
            for i, res in enumerate(miss["got"]):
                logger.info("  %d. ID: %s | Score (RRF): %.4f", i + 1, res["id"], res["score"])
        logger.info("------------------------------------------\n")

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / "hybrid_results.csv"
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["query_id", "query", "expected_id", "retrieved_ids", "top_score", "note", "evaluation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    logger.info("Детальные результаты сохранены в %s", results_path)


if __name__ == "__main__":
    main()