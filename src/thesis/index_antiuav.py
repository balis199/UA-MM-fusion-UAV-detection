"""Dataclass-based indexing for preprocessed Anti-UAV RGB-T frames and sequences.

Provides a dataclass based interface for creating a sqlite database of all frames in the preprocessed Anti-UAV RGB-T dataset and their related sequences.

3 separate tables are created in the database:
- `frames_rgb`: contains all rgb frames in the dataset, with their path, sequence, frame number and all label data (gt bb, drone exists, conditions)
- `frames_thermal`: contains all thermal frames in the dataset, with their path, sequence, frame number and all label data (gt bb, drone exists, conditions)
- `sequences`: contains all sequences in the dataset, with their path and number of frames
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from thesis.paths import ANTI_UAV_DATA_DIR
from thesis.utils.db import save_db

SPLITS: tuple[str, ...] = ("train", "val", "test")

# table name -> modality folder inside processed/<split>/<sequence>/
MODALITY_TABLES: dict[str, str] = {
    "frames_rgb": "visible",
    "frames_thermal": "infrared",
}

FRAME_FILE = re.compile(r"_frame_(\d+)\.png$")


@dataclass
class FrameRecord:
    """One row of a `frames_rgb` / `frames_thermal` table."""

    path: str
    sequence: str
    split: str
    frame_id: int
    gt_x: float
    gt_y: float
    gt_w: float
    gt_h: float
    drone_exists: int
    conditions: frozenset[str]

    def to_row(self, condition_columns: list[str]) -> dict:
        """Row dict with every condition one-hot encoded as its own 0/1 column."""
        row = {key: value for key, value in asdict(self).items() if key != "conditions"}
        row.update({column: int(column in self.conditions) for column in condition_columns})
        return row


@dataclass
class SequenceRecord:
    """One row of the `sequences` table."""

    sequence: str
    split: str
    path: str
    num_frames: int

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class IndexConfig:
    dataset_root: Path = ANTI_UAV_DATA_DIR
    db_name: str = "index.db"
    splits: tuple[str, ...] = SPLITS

    processed_dir: Path = field(init=False)
    conditions_dir: Path = field(init=False)
    db_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.processed_dir = self.dataset_root / "processed"
        self.conditions_dir = self.dataset_root / "label_new"
        self.db_path = self.dataset_root / self.db_name


def load_conditions(cfg: IndexConfig) -> tuple[dict[str, dict[str, frozenset[str]]], list[str]]:
    """Per split, map sequence id -> its condition tags.

    Also returns the sorted list of all condition tags seen, which become the
    one-hot encoded columns of the frames tables.
    """
    by_split: dict[str, dict[str, frozenset[str]]] = {}
    all_tags: set[str] = set()
    for split in cfg.splits:
        path = cfg.conditions_dir / f"{split}.json"
        if not path.exists():
            print(f"Conditions file {path} not found, sequences of {split!r} get no conditions")
            continue
        with open(path) as f:
            raw: dict[str, list[str]] = json.load(f)
        by_split[split] = {sequence: frozenset(tags) for sequence, tags in raw.items()}
        all_tags.update(tag for tags in raw.values() for tag in tags)
    return by_split, sorted(all_tags)


def load_frame_labels(raw_sequence_dir: Path, modality: str) -> tuple[list[int], list[list[float]]]:
    """Per-frame `exist` flags and `gt_rect` boxes of one raw sequence."""
    with open(raw_sequence_dir / f"{modality}.json") as f:
        labels = json.load(f)
    return labels["exist"], labels["gt_rect"]


def frame_records(
    cfg: IndexConfig,
    split: str,
    sequence: str,
    modality: str,
    conditions: frozenset[str],
) -> list[FrameRecord]:
    """One record per preprocessed frame of a sequence/modality pair.

    The frame number encoded in the filename indexes directly into the raw
    per-frame label arrays of the original video.
    """
    frame_dir = cfg.processed_dir / split / sequence / modality
    raw_dir = cfg.dataset_root / split / sequence
    exist, gt_rects = load_frame_labels(raw_dir, modality)

    records: list[FrameRecord] = []
    paths = sorted(frame_dir.iterdir(), key=lambda p: p.name)
    for path in tqdm(paths, desc=f"{split}/{sequence}/{modality}", unit="frame", leave=False):
        match = FRAME_FILE.search(path.name)
        if match is None:
            continue
        frame_id = int(match.group(1))
        if frame_id >= len(exist):
            raise IndexError(
                f"Frame {frame_id} of {split}/{sequence}/{modality} is not covered "
                f"by {raw_dir / (modality + '.json')} ({len(exist)} labels)"
            )
        if exist[frame_id] and len(gt_rects[frame_id]) == 4:
            gt_x, gt_y, gt_w, gt_h = gt_rects[frame_id]
        else:
            gt_x = gt_y = gt_w = gt_h = 0
        records.append(
            FrameRecord(
                path=path.relative_to(cfg.dataset_root).as_posix(),
                sequence=sequence,
                split=split,
                frame_id=frame_id,
                gt_x=gt_x,
                gt_y=gt_y,
                gt_w=gt_w,
                gt_h=gt_h,
                drone_exists=int(exist[frame_id]),
                conditions=conditions,
            )
        )
    return records


def index_dataset(cfg: IndexConfig) -> Path:
    """Walk the processed dataset and write the `frames_rgb`, `frames_thermal`
    and `sequences` tables to the sqlite database at ``cfg.db_path``."""
    frames: dict[str, list[FrameRecord]] = {table: [] for table in MODALITY_TABLES}
    sequences: list[SequenceRecord] = []
    conditions_by_split, condition_columns = load_conditions(cfg)
    print(f"One-hot encoding {len(condition_columns)} conditions: {condition_columns}")

    for split in cfg.splits:
        split_dir = cfg.processed_dir / split
        if not split_dir.is_dir():
            print(f"Split {split!r} not found under {cfg.processed_dir}, skipping")
            continue
        conditions = conditions_by_split.get(split, {})
        sequence_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        print(f"Split {split!r}: indexing {len(sequence_dirs)} sequences in {split_dir}")
        for sequence_dir in tqdm(sequence_dirs, desc=f"Indexing {split}", unit="sequence"):
            sequence = sequence_dir.name
            num_frames = 0
            for table, modality in MODALITY_TABLES.items():
                records = frame_records(
                    cfg, split, sequence, modality, conditions.get(sequence, frozenset())
                )
                frames[table].extend(records)
                # rgb and thermal are cut up with the same parameters, so the
                # first modality's frame count is the sequence's frame count
                if num_frames == 0:
                    num_frames = len(records)
            sequences.append(
                SequenceRecord(
                    sequence=sequence,
                    split=split,
                    path=sequence_dir.relative_to(cfg.dataset_root).as_posix(),
                    num_frames=num_frames,
                )
            )

    for table, records in frames.items():
        df = pd.DataFrame([record.to_row(condition_columns) for record in records])
        save_db(df, cfg.db_path, table)
        print(f"Saved {len(df)} rows to table {table!r} in {cfg.db_path}")
    sequences_df = pd.DataFrame([record.to_row() for record in sequences])
    save_db(sequences_df, cfg.db_path, "sequences")
    print(f"Saved {len(sequences_df)} rows to table 'sequences' in {cfg.db_path}")
    return cfg.db_path


if __name__ == "__main__":
    cfg = IndexConfig(dataset_root=ANTI_UAV_DATA_DIR)
    index_dataset(cfg)
