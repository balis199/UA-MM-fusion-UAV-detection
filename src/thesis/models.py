from pathlib import Path

import torch

from thesis.config import Config

# torch.hub entry point for YOLOv5. The master hubconf imports the `ultralytics`
# package and handles torch>=2.6 weights_only loading of pickled checkpoints,
# so `ultralytics` (plus yolov5's other hub requirements) must be installed -
# see the dev dependency group in pyproject.toml.
YOLOV5_HUB_REPO = "ultralytics/yolov5"


def load_model(cfg: Config, device: torch.device, train: bool) -> torch.nn.Module:

    if "yolo" in cfg.model_name:
        if cfg.model_yolo_weights is not None:
            # local checkpoint, e.g. model_weights/edtc_yolov5s.pt
            model = torch.hub.load(
                YOLOV5_HUB_REPO,
                "custom",
                path=str(Path(cfg.model_yolo_weights)),
                trust_repo=True,
            )
        else:
            model = torch.hub.load(
                YOLOV5_HUB_REPO,
                cfg.model_name,
                pretrained=cfg.model_yolo_pretrained,
                trust_repo=True,
            )

    elif "unet" in cfg.model_name:
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            cfg.model_name,
            in_channels=cfg.model_unet_in_channels,
            out_channels=cfg.model_unet_out_channels,
            init_features=cfg.model_unet_init_features,
            pretrained=cfg.model_unet_pretrained,
        )

    else:
        raise KeyError(f"Invalid model_name: {cfg.model_name}")

    assert isinstance(model, torch.nn.Module)

    model.to(device=device)
    if not train:
        model.eval()

    return model


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = Config(model_name="yolov5s")
    model = load_model(cfg, DEVICE, False)
    print(model)

    cfg = Config(model_name="unet")
    model = load_model(cfg, DEVICE, False)
    print(model)
