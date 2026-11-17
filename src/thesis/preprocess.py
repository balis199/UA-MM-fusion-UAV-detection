"""Dataclass-based preprocessing for Anti-UAV RGB-T sequences.

Two levels of abstraction:
- :class:`Sequence` — which frames belong together: cut up into folders,
  index single frames, and take time-shifted windows (RGB/thermal stay in sync
  because the shift is defined once per sequence).
- :class:`Frame` — what a single frame looks like: resize. Image operations
  return new ``Frame`` instances instead of mutating.

:class:`Preprocess` walks a dataset root (``<root>/<split>/<sequence_id>/``
with one video or image folder per modality) and cuts up every sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

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

    def save(self, out_dir: str | Path) -> Path:
        """Save the frame under ``out_dir`` with a self-describing filename."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.sequence_id}_{self.modality}_frame_{self.frame_id:06d}.png"
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
        return [self.frame(modality, start + shift + offset) for offset in range(length)]

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
        return [self.frame(modality, frame_id).save(out_dir) for frame_id in frame_ids]


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
                continue
            found.extend(Sequence(path) for path in sorted(split_dir.iterdir()) if path.is_dir())
        return found

    def process(
        self,
        modalities: tuple[str, ...] = ("visible", "infrared"),
        step: int = 1,
        shift: int = 0,
    ) -> list[Path]:
        """Cut up every sequence of every split into ``<output_dir>/<split>/<sequence>/<modality>/``."""
        saved: list[Path] = []
        for split in self.SPLITS:
            for sequence in self.sequences(split):
                for modality in modalities:
                    out_dir = self.output_dir / split / sequence.sequence_id / modality
                    try:
                        saved.extend(sequence.cutup(modality, out_dir, step=step, shift=shift))
                    except FileNotFoundError:
                        continue  # modality missing for this sequence
        return saved
