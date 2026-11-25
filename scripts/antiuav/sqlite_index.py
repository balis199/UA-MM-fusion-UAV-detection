"""Build the Anti-UAV sqlite index (index.db) from the processed frames + labels.

Walks <dataset_root>/processed/<split>/<sequence>/<modality> and the per-split
condition labels in <dataset_root>/label_new, then writes the frames_rgb,
frames_thermal and sequences tables. Run the preprocessing first:

    python scripts/antiuav/preprocess.py --step 10

Example:
    python scripts/antiuav/sqlite_index.py
    python scripts/antiuav/sqlite_index.py --splits train val
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis.antiuav.index import SPLITS, IndexConfig, index_dataset
from thesis.paths import ANTI_UAV_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=ANTI_UAV_DATA_DIR,
        help="Path to the Anti-UAV RGB-T dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--db-name", default="index.db", help="Index database name (default: %(default)s)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to index (default: all)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_root}")

    cfg = IndexConfig(dataset_root=dataset_root, db_name=args.db_name, splits=tuple(args.splits))
    db_path = index_dataset(cfg)
    print(f"Index written to {db_path}")


if __name__ == "__main__":
    main()
