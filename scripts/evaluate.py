"""Evaluate every trained model on validation and test and print the results table.

The best model is selected on **validation** NDCG@10; test metrics are reported once, for
information. Results are written to ``artifacts/results.json`` (used by the API and the
batch export to pick the serving model) and ``artifacts/results.md``.

Usage:
    python scripts/evaluate.py [--models popular mf two_tower sasrec]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from recsys.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_DIR,
    MODEL_ORDER,
    checkpoint_path,
    results_path,
)
from recsys.data import load_dataset
from recsys.evaluate import evaluate
from recsys.metrics import DEFAULT_KS
from recsys.models import MODEL_REGISTRY, load_model

SELECTION_METRIC = "NDCG@10"


def results_table(results: dict[str, dict[str, dict[str, float]]], stage: str) -> str:
    """Markdown table of ``stage`` metrics, one row per model."""
    columns = [f"{m}@{k}" for k in DEFAULT_KS for m in ("Recall", "NDCG")]
    lines = [
        "| Model | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    best = {c: max(r[stage][c] for r in results.values()) for c in columns}
    for name, result in results.items():
        cells = [
            f"**{result[stage][c]:.4f}**"
            if result[stage][c] == best[c]
            else f"{result[stage][c]:.4f}"
            for c in columns
        ]
        lines.append(f"| {MODEL_REGISTRY[name].display_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=MODEL_ORDER)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    data = load_dataset(args.data_dir)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name in args.models:
        path = checkpoint_path(args.artifacts_dir, name)
        if not path.exists():
            print(f"skip {name}: no checkpoint at {path}")
            continue
        model = load_model(path)
        results[name] = {stage: evaluate(model, data, stage) for stage in ("valid", "test")}
        print(f"evaluated {name}")
    if not results:
        raise SystemExit("No trained models found. Run `python scripts/train.py --model all`.")

    best_model = max(results, key=lambda name: results[name]["valid"][SELECTION_METRIC])
    payload = {
        "best_model": best_model,
        "selection": f"valid {SELECTION_METRIC}",
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_users": data.n_users,
        "n_items": data.n_items,
        "results": results,
    }
    out = results_path(args.artifacts_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = (
        "### Test (leave-one-out, full ranking)\n\n"
        + results_table(results, "test")
        + "\n\n### Validation\n\n"
        + results_table(results, "valid")
        + f"\n\nBest model by valid {SELECTION_METRIC}: **{best_model}**\n"
    )
    (args.artifacts_dir / "results.md").write_text(report, encoding="utf-8")
    print()
    print(report)


if __name__ == "__main__":
    main()
