"""Iterate a few batches of the Anti-UAV RGB-T DataLoader over one split.

The sqlite index must exist already: build it with "uv run python scripts/index_antiuav_script.py".

Example:
    python scripts/dataloader_antiuav_script.py --split val --batch-size 8 --max-batches 5
    python scripts/dataloader_antiuav_script.py --split test --conditions FM --drone-only
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thesis.dataset_antiuav import AntiUAVFrames
from thesis.index_antiuav import SPLITS
from thesis.paths import ANTI_UAV_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="train", help="Split to load")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers (default: 0)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="Stop after this many batches; 0 iterates the whole split (default: 5)",
    )
    parser.add_argument("--drone-only", action="store_true", help="Keep only frames with a drone")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        metavar="TAG",
        help="Keep only frames whose sequence has these condition tags; 'neutral' = no tags "
        "(e.g. --conditions FM LI, or --conditions neutral)",
    )
    parser.add_argument(
        "--condition-how",
        choices=["any", "all"],
        default="any",
        help="With --conditions: keep frames with any or all of the tags (default: any)",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        metavar="SEQ",
        help="Restrict to these sequence ids (e.g. --sequences 01739 01740)",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="Subsample this fraction of the split (default: 1.0)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed for --frac (default: 7)")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling")
    args = parser.parse_args()

    dataset = AntiUAVFrames(split=args.split, dataset_root=ANTI_UAV_DATA_DIR).view(
        conditions=args.conditions,
        how=args.condition_how,
        drone_only=True if args.drone_only else None,
        sequences=args.sequences,
        frac=args.frac,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
    )

    # overview before iterating
    split_drones = int(dataset.df["drone_exists"].sum())
    print(f"Split      : {args.split} ({len(dataset)} frame pairs, {split_drones} with drone)")
    print(f"Conditions : {dataset.condition_columns}")
    print(
        f"View       : conditions={args.conditions} (how={args.condition_how}), "
        f"sequences={args.sequences}, drone_only={args.drone_only}, frac={args.frac}"
    )
    print(
        f"Loader     : batch_size={args.batch_size}, shuffle={not args.no_shuffle}, "
        f"num_workers={args.num_workers}\n"
    )

    max_batches = args.max_batches or None  # 0 -> iterate the whole split
    total = max_batches if max_batches is not None else len(loader)
    seen = 0
    drones = 0
    batch_count = 0
    condition_sum = torch.zeros(len(dataset.condition_columns))
    box_wh_sum = torch.zeros(2)
    shapes = None
    rgb_min = rgb_max = thermal_min = thermal_max = None
    for batch in tqdm(loader, total=total, desc=f"{args.split} batches", unit="batch"):
        if max_batches is not None and batch_count >= max_batches:
            break
        batch_count += 1

        rgb, thermal = batch["rgb"], batch["thermal"]
        exists = batch["drone_exists"]
        seen += len(exists)
        drones += int(exists.sum())
        condition_sum += batch["conditions"].sum(dim=0)
        if shapes is None:
            shapes = (tuple(rgb.shape), tuple(thermal.shape))
        rgb_min = rgb.min().item() if rgb_min is None else min(rgb_min, rgb.min().item())
        rgb_max = rgb.max().item() if rgb_max is None else max(rgb_max, rgb.max().item())
        thermal_min = (
            thermal.min().item() if thermal_min is None else min(thermal_min, thermal.min().item())
        )
        thermal_max = (
            thermal.max().item() if thermal_max is None else max(thermal_max, thermal.max().item())
        )
        drone_boxes = batch["box_rgb"][exists == 1][:, 2:]  # w, h of drone frames
        if len(drone_boxes):
            box_wh_sum += drone_boxes.sum(dim=0)

    # summary after iterating
    print(
        f"\nIterated {batch_count} batches, {seen} samples: "
        f"{drones} with drone ({drones / max(seen, 1):.1%})"
    )
    if shapes is not None:
        print(
            f"rgb {shapes[0]} in [{rgb_min:.2f}, {rgb_max:.2f}] | "
            f"thermal {shapes[1]} in [{thermal_min:.2f}, {thermal_max:.2f}]"
        )
    if drones:
        mean_wh = (box_wh_sum / drones).tolist()
        print(f"mean box_rgb w,h over drone frames: [{mean_wh[0]:.1f}, {mean_wh[1]:.1f}]")
    top = condition_sum.argsort(descending=True)[:5]
    print(
        "Top conditions: "
        + ", ".join(f"{dataset.condition_columns[j]}={int(condition_sum[j])}" for j in top)
    )


if __name__ == "__main__":
    main()
