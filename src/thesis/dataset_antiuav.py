"""Main PyTorch Dataset over the Anti-UAV RGB-T sqlite index.

Joins the frames_rgb and frames_thermal tables written by thesis.index_antiuav
on (sequence, frame_id) (both modalities are cut up with the same parameters,
so they align 1:1) and yields one sample per RGB-T frame pair:

- rgb / thermal: transformed image tensors (thermal as RGB, or single-channel
  grayscale with thermal_channels=1),
- box_rgb / box_thermal: [4] float32 gt boxes (x, y, w, h) in the original pixel
  coordinates of the respective modality (zeros when no drone),
- drone_exists: scalar long,
- conditions: [num_conditions] float32 one-hot over AntiUAVFrames.condition_columns,
- sequence, frame_id, rgb_path, thermal_path: metadata.

Samples are plain dicts, so the default collate of torch.utils.data.DataLoader
batches them (tensors are stacked, strings become lists, ints become tensors).
Build the loader at the call site, e.g. DataLoader(dataset, batch_size=32).
The full index table stays available as dataset.df.

Sub-dataset views: `view(conditions=..., how="any"|"all", drone_only=...,
sequences=[...], frac=..., seed=...)` returns a lightweight chainable subset over the
same df without re-reading index.db; pass the virtual tag NEUTRAL ("neutral") in
`conditions` for frames whose sequence has no adverse tags at all. `with_transform(...)`
swaps the rgb/thermal transforms on a copy. The generic machinery lives in
thesis.utils.dataset.DataFrameDataset.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection
from pathlib import Path
from typing import Literal, Self, override

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from thesis.index_antiuav import SPLITS
from thesis.paths import ANTI_UAV_DATA_DIR, join
from thesis.transforms import TRANSFORM_DEFAULT
from thesis.utils.dataset import DataFrameDataset, SampleType
from thesis.utils.db import load_db

# columns every frames table has besides the one-hot condition columns
BASE_COLUMNS: tuple[str, ...] = (
    "path",
    "sequence",
    "split",
    "frame_id",
    "gt_x",
    "gt_y",
    "gt_w",
    "gt_h",
    "drone_exists",
)

# virtual condition usable in AntiUAVFrames.view(conditions=[...]): frames whose sequence
# has no adverse tags at all, i.e. none of the one-hot condition columns is 1
NEUTRAL = "neutral"


class AntiUAVFrames(DataFrameDataset):
    """Temporally Aligned RGB-T frame pairs of one split, indexed by the sqlite database.

    Args:
        split: One of "train", "val", "test".
        dataset_root: Anti-UAV RGB-T dataset root holding the index database.
        db_name: Name of the sqlite index inside dataset_root.
        transform: Torchvision v2 transform applied to the rgb image.
        thermal_transform: Transform for the thermal image (defaults to transform).
        thermal_channels: 3 to load thermal as RGB, 1 for single-channel grayscale.
        drone_only: Keep only frames with drone_exists == 1.
        condition_columns: Subset of the index's one-hot condition columns to
            return (default: all of them).
        frac: Keep only this fraction of the frames (after filtering).
        seed: Random state for frac subsampling.
    """

    def __init__(
        self,
        split: str = "train",
        dataset_root: Path | str = ANTI_UAV_DATA_DIR,
        db_name: str = "index.db",
        transform: v2.Transform = TRANSFORM_DEFAULT,
        thermal_transform: v2.Transform | None = None,
        thermal_channels: int = 3,
        drone_only: bool = False,
        condition_columns: list[str] | None = None,
        frac: float = 1.0,
        seed: int = 7,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}")
        if thermal_channels not in (1, 3):
            raise ValueError("thermal_channels must be 1 (grayscale) or 3 (RGB)")
        if not 0 < frac <= 1.0:
            raise ValueError("frac must be in (0, 1]")

        self.dataset_root = Path(dataset_root)
        db_path = self.dataset_root / db_name
        if not db_path.exists():
            raise FileNotFoundError(
                f"Index database not found: {db_path}. "
                'Build it first with "uv run python scripts/index_antiuav_script.py"'
            )
        self.transform = transform
        self.thermal_transform = thermal_transform if thermal_transform is not None else transform
        self.thermal_channels = thermal_channels

        rgb = load_db(db_path, "frames_rgb")
        thermal = load_db(db_path, "frames_thermal")
        rgb = rgb[rgb["split"] == split].rename(columns={"path": "rgb_path"})
        # keep the thermal gt boxes too: they live in the thermal image's own pixel
        # coordinates (e.g. 640x512 vs 1920x1080 for rgb)
        thermal = thermal[thermal["split"] == split][
            ["sequence", "frame_id", "path", "gt_x", "gt_y", "gt_w", "gt_h"]
        ].rename(
            columns={
                "path": "thermal_path",
                "gt_x": "gt_x_thermal",
                "gt_y": "gt_y_thermal",
                "gt_w": "gt_w_thermal",
                "gt_h": "gt_h_thermal",
            }
        )
        # rgb and thermal are cut up with the same parameters -> aligned on (sequence, frame_id)
        df = rgb.merge(thermal, on=["sequence", "frame_id"], validate="one_to_one")

        if condition_columns is None:
            skip = (
                *BASE_COLUMNS,
                "rgb_path",
                "thermal_path",
                "gt_x_thermal",
                "gt_y_thermal",
                "gt_w_thermal",
                "gt_h_thermal",
            )
            condition_columns = [c for c in df.columns if c not in skip]
        missing = [c for c in condition_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Condition columns not present in the index: {missing}")
        self.condition_columns = list(condition_columns)

        if drone_only:
            df = df[df["drone_exists"] == 1].reset_index(drop=True)
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
        super().__init__(df)

    @override
    def view(
        self,
        *,
        mask: pd.Series | None = None,
        conditions: Collection[str] | None = None,
        how: Literal["any", "all"] = "any",
        drone_only: bool | None = None,
        sequences: Collection[str] | None = None,
        frac: float = 1.0,
        seed: int = 7,
    ) -> Self:
        """Anti-UAV filter sugar on top of DataFrameDataset.view.

        Args:
            mask: Extra boolean Series aligned with self.df, ANDed with the filter
                masks below (the generic DataFrameDataset.view argument).
            conditions: Condition tags; keeps frames whose sequence carries them. The
                virtual tag NEUTRAL ("neutral") selects frames whose sequence has no tags
                at all (none of the condition columns == 1), e.g. conditions=["neutral"].
                how="all" with NEUTRAL plus any other condition always yields an empty
                view, so a warning is issued for that combination.
            how: "any" keeps frames matching at least one of the conditions, "all" requires all.
            drone_only: Additionally keep only frames with drone_exists == 1;
                None keeps the parent's current filtering.
            sequences: Additionally restrict to these sequence ids.
            frac: Keep only this fraction of the filtered frames.
            seed: Random state for frac subsampling.
        """
        if how not in ("any", "all"):
            raise ValueError(f'how must be "any" or "all", got {how!r}')
        if conditions is not None and len(conditions) == 0:
            raise ValueError("conditions=[] selects nothing; use conditions=[NEUTRAL] for no-tags")
        if (
            conditions is not None
            and how == "all"
            and NEUTRAL in conditions
            and len(conditions) > 1
        ):
            warnings.warn(
                f'how="all" with NEUTRAL and other conditions {list(conditions)} always '
                "returns an empty view (a frame cannot be both neutral and tagged)",
                stacklevel=2,
            )

        filters: pd.Series | None = None
        if conditions is not None:
            valid = [*self.condition_columns, NEUTRAL]
            unknown = [c for c in conditions if c not in valid]
            if unknown:
                raise ValueError(f"Unknown conditions {unknown}; choose from {valid}")
            neutral_mask = self.df[self.condition_columns].sum(axis=1) == 0
            masks = [neutral_mask if c == NEUTRAL else self.df[c] == 1 for c in conditions]
            filters = masks[0]
            for m in masks[1:]:
                filters = filters | m if how == "any" else filters & m
        if sequences is not None:
            seq_mask = self.df["sequence"].isin(list(sequences))
            filters = seq_mask if filters is None else filters & seq_mask
        if drone_only:
            drone_mask = self.df["drone_exists"] == 1
            filters = drone_mask if filters is None else filters & drone_mask
        if mask is not None:
            filters = mask if filters is None else filters & mask
        return super().view(mask=filters, frac=frac, seed=seed)

    @override
    def with_transform(
        self, transform: v2.Transform, thermal_transform: v2.Transform | None = None
    ) -> Self:
        """Copy with swapped rgb/thermal transforms; shares the df (see DataFrameDataset)."""
        new = super().with_transform(transform)
        new.thermal_transform = thermal_transform if thermal_transform is not None else transform
        return new

    def _load_image(self, relative: str, channels: int) -> Image.Image:
        # index paths may contain Windows separators; join() normalizes them on any OS
        image = Image.open(join(self.dataset_root, relative))
        return image.convert("L" if channels == 1 else "RGB")

    @override
    def __getitem__(self, index: int) -> SampleType:
        # parameter name must stay "index" to match torch.utils.data.Dataset.__getitem__
        row = self.df.iloc[index]
        return {
            "rgb": self.transform(self._load_image(str(row["rgb_path"]), 3)),
            "thermal": self.thermal_transform(
                self._load_image(str(row["thermal_path"]), self.thermal_channels)
            ),
            "box_rgb": torch.tensor(
                [row["gt_x"], row["gt_y"], row["gt_w"], row["gt_h"]], dtype=torch.float32
            ),
            "box_thermal": torch.tensor(
                [
                    row["gt_x_thermal"],
                    row["gt_y_thermal"],
                    row["gt_w_thermal"],
                    row["gt_h_thermal"],
                ],
                dtype=torch.float32,
            ),
            "drone_exists": torch.tensor(int(row["drone_exists"]), dtype=torch.long),
            "conditions": torch.tensor(
                [int(row[c]) for c in self.condition_columns], dtype=torch.float32
            ),
            "sequence": str(row["sequence"]),
            "frame_id": int(row["frame_id"]),
            "rgb_path": str(row["rgb_path"]),
            "thermal_path": str(row["thermal_path"]),
        }
