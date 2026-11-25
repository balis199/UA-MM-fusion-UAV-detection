"""Feature extraction over the Anti-UAV RGB-T frame pairs (work in progress).

Iterates one split of `AntiUAVFrames` (sqlite index -> aligned RGB-T frame pairs), runs a
model from `thesis.models.load_model` on one modality, post-processes the raw YOLO output
with `NMSYoloBoxes`, and saves the features to an npz file (key 'data') under
`cfg.output_dir`.

Run:
    uv run python scripts/antiuav/extract_features.py
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thesis.antiuav.dataset import AntiUAVFrames
from thesis.config import Config
from thesis.models import load_model
from thesis.paths import ANTI_UAV_DATA_DIR
from thesis.transforms import NMSYoloBoxes

postproc_nms = NMSYoloBoxes(0.3, 0.3)


def extract_features(cfg: Config, device: torch.device, modality: str = "rgb") -> Path:
    """Run the model over one split and save the post-NMS features to an npz file."""
    if modality not in ("rgb", "thermal"):
        raise ValueError(f"modality must be 'rgb' or 'thermal', got {modality!r}")

    dataset = AntiUAVFrames(
        split=cfg.data_split,
        dataset_root=ANTI_UAV_DATA_DIR,
        frac=cfg.data_frac,
        seed=cfg.data_seed,
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=cfg.data_batch_size)
    model = load_model(cfg, device, train=False)

    data = []
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Extracting {modality}")
        for batch in pbar:
            images = batch[modality].to(device)
            out = model(images)
            out = postproc_nms(out)
            data.append(out)

    data = torch.cat(data, dim=0).cpu().numpy()
    out_path = Path(cfg.output_dir) / f"features_{cfg.model_name}_{modality}_{cfg.data_split}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, data=data)
    print(f"{data.shape} -> {out_path}")
    return out_path
