"""Dataclass-based preprocessing for Anti-UAV RGB-T sequences.

Two levels of abstraction:
- :class:`Sequence`: which frames belong together (cut up into folders,
  index single frames, and take time-shifted windows; RGB/thermal stay in sync
  because the shift is defined once per sequence).
- :class:`Frame`: what a single frame looks like (resize). Image operations
  return new ``Frame`` instances instead of mutating.

:class:`Preprocess` walks a dataset root (``<root>/<split>/<sequence_id>/``
with one video or image folder per modality) and cuts up every sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Frame:
    """A single frame of a sequence with per-frame image operations."""

    image: np.ndarray
    frame_id: int
    sequence_id: str
    modality: str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape

    def resize(self, size: tuple[int, int]) -> Frame:
        """Return a resized copy; ``size`` is (width, height) as in cv2."""
        resized = cv2.resize(self.image, size, interpolation=cv2.INTER_AREA)
        return replace(self, image=resized)

    @staticmethod
    def filename(sequence_id: str, modality: str, frame_id: int) -> str:
        """Self-describing filename for a saved frame."""
        return f"{sequence_id}_{modality}_frame_{frame_id:06d}.png"

    def save(self, out_dir: str | Path) -> Path:
        """Save the frame under ``out_dir`` with a self-describing filename."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / self.filename(self.sequence_id, self.modality, self.frame_id)
        if not cv2.imwrite(str(path), self.image):
            raise OSError(f"Could not save frame to {path}")
        return path


@dataclass
class Sequence:
    """A sequence directory holding one video or image folder per modality."""

    path: Path
    sequence_id: str | None = None

    modality_files: ClassVar[dict[str, str]] = {
        "visible": "visible.mp4",
        "infrared": "infrared.mp4",
    }

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.sequence_id is None:
            self.sequence_id = self.path.name

    def source(self, modality: str) -> Path:
        """Path to the video file or image folder of a modality."""
        if modality not in self.modality_files:
            raise ValueError(
                f"Unknown modality {modality!r}; choose from {sorted(self.modality_files)}"
            )
        candidate = self.path / self.modality_files[modality]
        if candidate.exists():
            return candidate
        image_dir = self.path / modality
        if image_dir.is_dir():
            return image_dir
        raise FileNotFoundError(f"No source for modality {modality!r} in {self.path}")

    def _frame_paths(self, source: Path) -> list[Path]:
        """Return image frames in deterministic filename order."""
        return sorted(
            (
                path
                for path in source.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name,
        )

    def frame_count(self, modality: str) -> int:
        """Number of frames available for a modality."""
        source = self.source(modality)
        if source.is_dir():
            return len(self._frame_paths(source))
        capture = cv2.VideoCapture(str(source))
        try:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

    def frame(self, modality: str, frame_id: int) -> Frame:
        """Extract a zero-based frame from the modality's video or image folder."""
        if frame_id < 0:
            raise ValueError("frame_id must be non-negative")

        source = self.source(modality)
        if source.is_dir():
            frames = self._frame_paths(source)
            if frame_id >= len(frames):
                raise IndexError(f"Frame {frame_id} is unavailable in {source}")
            image = cv2.imread(str(frames[frame_id]), cv2.IMREAD_UNCHANGED)
        else:
            capture = cv2.VideoCapture(str(source))
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                success, image = capture.read()
            finally:
                capture.release()
            if not success:
                raise IndexError(f"Frame {frame_id} is unavailable in {source}")

        if image is None:
            raise OSError(f"Could not read frame {frame_id} from {source}")
        return Frame(
            image=image, frame_id=frame_id, sequence_id=self.sequence_id, modality=modality
        )

    def window(self, modality: str, start: int, length: int, shift: int = 0) -> list[Frame]:
        """Time-shifted window of frames at indices ``start+shift .. start+shift+length-1``."""
        if length < 1:
            raise ValueError("length must be positive")
        offsets = tqdm(range(length), desc=f"{self.sequence_id}/{modality} window", unit="frame")
        return [self.frame(modality, start + shift + offset) for offset in offsets]

    def cutup(
        self,
        modality: str,
        out_dir: str | Path,
        start: int = 0,
        step: int = 1,
        shift: int = 0,
    ) -> list[Path]:
        """Extract frames (optionally time-shifted and strided) and save them into a folder."""
        if step < 1:
            raise ValueError("step must be positive")
        count = self.frame_count(modality)
        frame_ids = range(start + shift, count, step)
        print(
            f"{self.sequence_id}/{modality}: cutting up {len(frame_ids)} frames "
            f"(start={start}, step={step}, shift={shift})"
        )
        frames = tqdm(
            frame_ids, total=len(frame_ids), desc=f"{self.sequence_id}/{modality}", unit="frame"
        )
        saved: list[Path] = []
        for frame_id in frames:
            saved.append(self.frame(modality, frame_id).save(out_dir))
        print(f"{self.sequence_id}/{modality}: saved {len(saved)} frames to {out_dir}")
        return saved

    def expected_paths(
        self,
        modality: str,
        out_dir: str | Path,
        start: int = 0,
        step: int = 1,
        shift: int = 0,
    ) -> list[Path]:
        """Paths :meth:`cutup` would write for these parameters, without writing them."""
        count = self.frame_count(modality)
        out = Path(out_dir)
        return [
            out / Frame.filename(self.sequence_id, modality, frame_id)
            for frame_id in range(start + shift, count, step)
        ]

    def is_processed(
        self,
        modality: str,
        out_dir: str | Path,
        start: int = 0,
        step: int = 1,
        shift: int = 0,
    ) -> bool:
        """True if every frame :meth:`cutup` would write already exists in ``out_dir``."""
        expected = self.expected_paths(modality, out_dir, start=start, step=step, shift=shift)
        return bool(expected) and all(path.exists() for path in expected)


class Preprocess:
    """Walk an Anti-UAV RGB-T dataset root and cut up every sequence."""

    SPLITS: ClassVar[tuple[str, ...]] = ("train", "val", "test")

    def __init__(self, dataset_root: str | Path, output_dir: str | Path | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.output_dir = (
            Path(output_dir) if output_dir is not None else self.dataset_root / "processed"
        )

    def sequences(self, split: str | None = None) -> list[Sequence]:
        """All sequence directories of one split, or of all splits."""
        splits = (split,) if split is not None else self.SPLITS
        found: list[Sequence] = []
        for name in splits:
            split_dir = self.dataset_root / name
            if not split_dir.is_dir():
                print(f"Split {name!r} not found under {self.dataset_root}, skipping")
                continue
            sequences = [Sequence(p) for p in sorted(split_dir.iterdir()) if p.is_dir()]
            print(f"Split {name!r}: found {len(sequences)} sequences in {split_dir}")
            found.extend(sequences)
        return found

    def process(
        self,
        modalities: tuple[str, ...] = ("visible", "infrared"),
        step: int = 1,
        shift: int = 0,
        skip_existing: bool = True,
    ) -> list[Path]:
        """Cut up every sequence of every split into ``<output_dir>/<split>/<sequence>/<modality>/``.

        With ``skip_existing`` (the default), pairs whose expected frame files already
        exist in the output are skipped (reported per sequence), so interrupted runs resume.
        """
        tasks = [
            (split, sequence, modality)
            for split in self.SPLITS
            for sequence in self.sequences(split)
            for modality in modalities
        ]
        print(
            f"Cutting up {len(tasks)} sequence/modality pairs "
            f"(step={step}, shift={shift}) into {self.output_dir}"
        )
        saved: list[Path] = []
        skipped = 0
        with tqdm(total=len(tasks), desc="Processing", unit="pair") as pbar:
            for (split, sequence), group in groupby(tasks, key=lambda task: (task[0], task[1])):
                done: list[str] = []
                missing: list[str] = []
                for _, _, modality in group:
                    pbar.set_description(f"{split}/{sequence.sequence_id}/{modality}")
                    pbar.update(1)
                    out_dir = self.output_dir / split / sequence.sequence_id / modality
                    try:
                        if skip_existing and sequence.is_processed(
                            modality, out_dir, step=step, shift=shift
                        ):
                            skipped += 1
                            done.append(modality)
                            continue
                        saved.extend(sequence.cutup(modality, out_dir, step=step, shift=shift))
                    except FileNotFoundError:
                        missing.append(modality)
                if done or missing:
                    reasons = []
                    if done:
                        reasons.append(f"already processed: {', '.join(done)}")
                    if missing:
                        reasons.append(f"no source: {', '.join(missing)}")
                    print(f"Skipping {split}/{sequence.sequence_id} ({'; '.join(reasons)})")
        if skip_existing:
            print(f"Saved {len(saved)} frames into {self.output_dir} ({skipped} already processed)")
        else:
            print(f"Saved {len(saved)} frames into {self.output_dir}")
        return saved
