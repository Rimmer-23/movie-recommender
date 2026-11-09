"""Download MovieLens 1M, preprocess it and print dataset statistics.

Usage:
    python scripts/download_data.py [--data-dir data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.data import (
    ML1M_URL,
    build_split,
    dataset_stats,
    download_movielens,
    preprocess,
    read_raw_movielens,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--url", default=ML1M_URL)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    ml_dir = download_movielens(args.data_dir / "raw", url=args.url, force=args.force)
    print(f"Raw data:       {ml_dir}")

    processed = preprocess(*read_raw_movielens(ml_dir))
    processed_dir = args.data_dir / "processed"
    processed.save(processed_dir)
    print(f"Processed data: {processed_dir}\n")

    split = build_split(processed)
    assert split.n_users == processed.users.shape[0]

    print("Dataset statistics")
    print("------------------")
    for key, value in dataset_stats(processed).items():
        print(f"{key:<20} {_format(value)}")


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


if __name__ == "__main__":
    main()
