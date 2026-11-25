"""Generic pandas-backed torch Dataset with chainable views and transform swaps.

Base class for the dataset-specific datasets (e.g. thesis.dataset_antiuav.AntiUAVFrames),
so the view/transform machinery is written once and reused across datasets:

- `view(mask=..., frac=..., seed=...)` returns a lightweight copy over a filtered df
  (views narrow, never widen; chainable; no re-reading of any index/database),
- `with_transform(transform)` returns a shallow copy with `transform` swapped while
  sharing the df, so different views of the same frames can use different transforms.

Subclasses store the index slice in `self.df`, set their transform attribute(s) in
`__init__`, implement `__getitem__`, and may override `view()` to add dataset-specific
filter sugar (see AntiUAVFrames for its conditions/sequences filters).
"""

from __future__ import annotations

import copy
from typing import Any, Self

import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2

SampleType = dict[str, Any]


class DataFrameDataset(Dataset[SampleType]):
    """A torch Dataset over a pandas DataFrame slice.

    Args:
        df: The index slice this dataset iterates over (kept as `self.df`).
    """

    df: pd.DataFrame
    transform: v2.Transform

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def view(
        self,
        *,
        mask: pd.Series | None = None,
        frac: float = 1.0,
        seed: int = 7,
    ) -> Self:
        """Return a lightweight subset over the current df.

        Args:
            mask: Boolean Series aligned with `self.df`; None keeps all rows.
            frac: Keep only this fraction of the (masked) rows.
            seed: Random state for frac subsampling.

        Views narrow, never widen: the mask is applied on top of the current df, so a
        view of a view only shrinks further. To widen, build a fresh dataset.
        """
        if not 0 < frac <= 1.0:
            raise ValueError("frac must be in (0, 1]")
        df = self.df if mask is None else self.df[mask]
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=seed)
        new = copy.copy(self)
        new.df = df.reset_index(drop=True)
        return new

    def with_transform(self, transform: v2.Transform) -> Self:
        """Return a shallow copy with `transform` swapped; shares the df.

        Subclasses with several named transforms (e.g. rgb + thermal) should override
        this and swap the remaining transforms themselves.
        """
        new = copy.copy(self)
        new.transform = transform
        return new
