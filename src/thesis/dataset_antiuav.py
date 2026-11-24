"""Main PyTorch Dataset over the Anti-UAV RGB-T sqlite index.

Joins the frames rgb and frames thermal tables written by thesis.index_antiuav
on (sequence, frame_id) - both modalities are cut up with the same parameters,
so they align 1:1 - and yields one sample per RGB-T frame pair:

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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from thesis.index_antiuav import SPLITS
from thesis.paths import ANTI_UAV_DATA_DIR
from thesis.transforms import TRANSFORM_DEFAULT
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

SampleType = dict[str, Any]


class AntiUAVFrames(Dataset[SampleType]):
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
                f"Index database not found: {db_path} - "
                'build it first with "python src/thesis/index_antiuav.py"'
            )
        self.transform = transform
        self.thermal_transform = thermal_transform if thermal_transform is not None else transform
        self.thermal_channels = thermal_channels

        rgb = load_db(db_path, "frames rgb")
        thermal = load_db(db_path, "frames thermal")
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
        self.df = df

    @override
    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, relative: str, channels: int) -> Image.Image:
        # index paths may contain Windows separators; forward slashes work everywhere
        image = Image.open(self.dataset_root / relative.replace("\\", "/"))
        return image.convert("L" if channels == 1 else "RGB")

    @override
    def __getitem__(self, idx: int) -> SampleType:
        row = self.df.iloc[idx]
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


if __name__ == "__main__":
    dataset = AntiUAVFrames(split="train")
    print(f"{len(dataset)} frame pairs, conditions: {dataset.condition_columns}")
    sample = dataset[0]
    print(f"{sample['sequence']} frame {sample['frame_id']}")
    print(
        f"rgb {tuple(sample['rgb'].shape)}, thermal {tuple(sample['thermal'].shape)}, "
        f"box_rgb {sample['box_rgb'].tolist()}, box_thermal {sample['box_thermal'].tolist()}, "
        f"drone_exists {int(sample['drone_exists'])}, conditions {sample['conditions'].tolist()}"
    )
    print(f"paths: {sample['rgb_path']} <-> {sample['thermal_path']}")
