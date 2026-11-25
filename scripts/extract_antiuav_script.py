"""Run feature extraction over one split of the Anti-UAV RGB-T frame pairs.

Iterates a split of `AntiUAVFrames` (see scripts/dataloader_antiuav_script.py for the
index it reads), runs the model on one modality, post-processes the raw YOLO output
with NMSYoloBoxes and saves the features to an npz under --output-dir. Details in
thesis.extract_antiuav.extract_features.

Example:
    python scripts/extract_antiuav_script.py --split train --modality rgb --batch-size 32
    python scripts/extract_antiuav_script.py --modality thermal --weights model_weights/edtc_yolov5s.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from thesis.config import Config
from thesis.extract_antiuav import extract_features
from thesis.index_antiuav import SPLITS
from thesis.paths import OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name", default="yolov5s", help="torch.hub model name (default: %(default)s)"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Local checkpoint to load via hub 'custom' (e.g. model_weights/edtc_yolov5s.pt)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't load pretrained hub weights (only with a hub model name)",
    )
    parser.add_argument("--modality", choices=["rgb", "thermal"], default="rgb")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--frac", type=float, default=1.0, help="Subsample the split")
    parser.add_argument("--seed", type=int, default=7, help="Seed for --frac")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = Config(
        model_name=args.model_name,
        model_yolo_pretrained=not args.no_pretrained,
        model_yolo_weights=args.weights,
        data_split=args.split,
        data_frac=args.frac,
        data_seed=args.seed,
        data_batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    extract_features(cfg, device, modality=args.modality)


if __name__ == "__main__":
    main()
