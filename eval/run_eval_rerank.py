"""
eval/run_eval_rerank.py — оценка качества поиска с применением реранкинга.
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
from src.reranker import Reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    eval_path = Path(settings.data_dir) / "eval" / "eval.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    retriever = Retriever()
    reranker = Reranker()

    normal_total = 0
    normal_hits_1 = 0  # Hit@1 (строгая метрика для реранкера)
    normal_hits_3 = 0  # Hit@3

    for item in eval_data:
        if item.get("note") != "should_match":
            continue

        query = item.get("description")
        expected_id = item.get("matches_index_id")

        if not query or not expected_id:
            continue

        normal_total += 1

        # 1. Получаем топ-10 кандидатов от гибридного поиска
        candidates = retriever.search(query, k=10)

        # 2. Реранжим их до топ-3
        final_results = reranker.rerank(query, candidates, top_k=3)

        retrieved_ids = [res["id"] for res in final_results]

        if expected_id in retrieved_ids:
            normal_hits_3 += 1
            if retrieved_ids[0] == expected_id:  # Если на самом первом месте
                normal_hits_1 += 1

    hit_rate_1 = (normal_hits_1 / normal_total * 100) if normal_total > 0 else 0.0
    hit_rate_3 = (normal_hits_3 / normal_total * 100) if normal_total > 0 else 0.0

    logger.info("=" * 60)
    logger.info("RERANKER EVAL RESULTS")
    logger.info("-" * 60)
    logger.info("Total queries: %d", normal_total)
    logger.info("Hit Rate @ 1: %.2f%% (Идеальный первый ответ)", hit_rate_1)
    logger.info("Hit Rate @ 3: %.2f%%", hit_rate_3)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
