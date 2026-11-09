"""Batch export: top-k recommendations for every user to a parquet table.

Simulates delivering an enriched table to a data warehouse. Each row is one
recommendation with the schema
``user_id, rank, movie_id, title, score, model, generated_at``.

Usage:
    python scripts/export_recommendations.py [--k 10] [--model sasrec]
        [--output artifacts/recommendations.parquet]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from recsys.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR
from recsys.serving import RecommendationService, write_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-k recommendations for all users.")
    parser.add_argument("--k", type=int, default=10, help="recommendations per user")
    parser.add_argument("--model", help="model to use (default: best by validation NDCG@10)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, help="default: <artifacts>/recommendations.parquet")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    start = time.perf_counter()
    service = RecommendationService.from_artifacts(
        artifacts_dir=args.artifacts_dir, data_dir=args.data_dir, model_name=args.model
    )
    table = service.recommend_all(k=args.k, batch_size=args.batch_size)
    output = args.output or args.artifacts_dir / "recommendations.parquet"
    write_recommendations(table, output)

    print(f"Model:   {service.model_name}")
    print(f"Rows:    {table.num_rows:,} ({service.n_users:,} users x top-{args.k})")
    print(f"Schema:  {', '.join(f'{f.name}: {f.type}' for f in table.schema)}")
    print(f"Written: {output} in {time.perf_counter() - start:.1f}s\n")
    print(table.slice(0, 5).to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
