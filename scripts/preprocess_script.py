"""Run the project's preprocessing pipeline for the Anti-UAV RGB-T dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis.paths import ANTI_UAV_DATA_DIR
from thesis.preprocess import Preprocess


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
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write cut-up frames (default: <dataset_root>/processed)",
    )
    parser.add_argument("--step", type=int, default=1, help="Keep every Nth frame (default: 1)")
    parser.add_argument(
        "--shift",
        type=int,
        default=0,
        help="Time-shift all windows by this many frames (default: 0)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_root}")

    saved = Preprocess(dataset_root, args.output_dir).process(step=args.step, shift=args.shift)
    print(f"Saved {len(saved)} frames under {args.output_dir or dataset_root / 'processed'}")


if __name__ == "__main__":
    main()
