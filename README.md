# UA-MM-fusion-UAV-detection
A repo that explores Uncertainty Aware Multimodal Fusion for UAV detection in adverse and out-of-distribution conditions.


Project structure adapted from am00ma/thesis-template.

## Current status

- **Data pipeline is done**: the Anti-UAV RGB-T videos are cut up into per-frame PNGs (`data/Anti-UAV-RGBT/processed/`) and indexed into a SQLite database (`index.db`) that maps every frame to its ground-truth box, drone-presence flag and adverse-condition tags.
- **A PyTorch dataloader now exists**: `src/thesis/dataset_antiuav.py` (`AntiUAVFrames`) joins the `frames_rgb` / `frames_thermal` index tables on `(sequence, frame_id)` and yields aligned RGB-T frame-pair samples (images, per-modality GT boxes, drone flag, condition one-hots). `scripts/dataloader_antiuav_script.py` iterates batches of it as a smoke test.
- **Modality alignment was explored and ruled out**: the RGB and IR ground-truth boxes themselves disagree, so pixel-level alignment of the two streams is not feasible to rely on. Current direction is **unaligned fusion**; "disagreement uncertainty" cannot be separated from GT disagreement, so it is not very explorable.
- **Model side is in early stages**: YOLOv5/UNet loading via `torch.hub` (`models.py`) and a feature-extraction loop over the Anti-UAV frames (`extract_antiuav.py`) exist, plus a YOLOv5s checkpoint (`model_weights/edtc_yolov5s.pt`) and `notebooks/predict_thermal_edtc_antiuav.ipynb` with first thermal predictions from that checkpoint. There is no fusion model, training loop or evaluation yet (`eval.py` and `metrics.py` are placeholders).

## Repository layout

```
├── TODO.md                             # prioritized next steps (config decision first)
├── config/default.json                 # default run configuration (not loaded into Config yet; see TODO.md)
├── model_weights/                      # model checkpoints (untracked)
│   └── edtc_yolov5s.pt                 #   YOLOv5s DetectionModel weights (~16 MB)
├── data/Anti-UAV-RGBT/                 # dataset root (gitignored, ~30 GB, NOT in the repo)
│   ├── train/ val/ test/               # 160 / 67 / 91 sequences of visible.mp4/json + infrared.mp4/json
│   ├── label_new/                      # <split>.json: sequence -> adverse-condition tags
│   ├── processed/                      # cut-up frames: <split>/<seq>/{visible,infrared}/*.png
│   ├── index.db                        # SQLite index: tables [frames_rgb], [frames_thermal], [sequences]
│   ├── datasette-metadata.yml          # canned queries for the datasette UI
│   └── framecut.py                     # original dataset video->jpg helper (not used by the pipeline)
├── experiments/ logs/                  # reserved for run outputs (kept via .gitkeep, contents gitignored)
├── notebooks/
│   ├── check_antiuav.ipynb             # sanity check: show one RGB/IR frame + GT box per modality
│   ├── check_alignment_antiuav.ipynb   # RGB/IR alignment: naive overlay, GT-center affine, train-only regression
│   ├── check_alignment_interactive_antiuav.py   # marimo app: global transforms (reg/radial/fisheye/+wh), IoU & AP on test
│   ├── check_alignment_per_sequence_antiuav.py  # marimo app: per-sequence fits with outlier handling (none/mad/ransac)
│   └── predict_thermal_edtc_antiuav.ipynb  # first thermal predictions with edtc_yolov5s.pt
├── scripts/
│   ├── preprocess_antiuav_script.py    # CLI wrapper around the preprocessing pipeline
│   ├── index_antiuav_script.py         # CLI wrapper: build index.db from processed frames + labels
│   ├── datasette_script.py             # serve index.db + frame thumbnails in datasette
│   ├── dataloader_antiuav_script.py    # CLI: iterate DataLoader batches of AntiUAVFrames over one split
│   └── extract_antiuav_script.py       # CLI wrapper: feature extraction over a split (AntiUAVFrames + model)
└── src/thesis/
    ├── paths.py                        # repo path constants (DATA_DIR, OUTPUT_DIR, ANTI_UAV_DATA_DIR, ...)
    ├── config.py                       # Config dataclass: data + model (yolo/unet) settings
    ├── preprocess_antiuav.py           # Frame / Sequence / Preprocess: video -> cut-up frames (resumable)
    ├── index_antiuav.py                # build index.db from processed frames + raw labels + conditions
    ├── dataset_antiuav.py              # AntiUAVFrames: torch Dataset of aligned RGB-T frame pairs over index.db
    ├── transforms.py                   # default image transform + YOLO NMS box post-processing (sample for custom transforms)
    ├── models.py                       # load_model(): yolov5 (ultralytics) / unet (brain-segmentation-pytorch) via torch.hub
    ├── extract_antiuav.py              # feature-extraction loop over AntiUAVFrames (work in progress)
    ├── eval.py, metrics.py             # placeholders for the upcoming evaluation
    └── utils/                          # sqlite helpers (db.py), generic DataFrameDataset base with views
                                        #   (dataset.py), git status helper (git.py), json helpers (json_io.py)
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/) (`.python-version` pins 3.13):

```bash
uv sync        # creates .venv, installs the `thesis` package (editable, src layout) + dependencies
```

The distribution is named `ua-mm-fusion-uav-detection`, but the import package stays `thesis` (`src/thesis`) via `[tool.uv.build-backend] module-name = "thesis"` in `pyproject.toml`.

Runtime dependencies (what `src/thesis` actually imports): torch, torchvision, opencv-python, numpy, pandas, Pillow, tqdm.
The dev group adds the notebook/exploration tooling: jupyter, ipykernel, marimo, matplotlib, scikit-learn, distinctipy, pyyaml, datasette + datasette-render-image-tags, and ruff (lint/format, line-length 100, configured in `pyproject.toml`).
`.vscode/settings.json` enables ruff format-on-save. Ruff is configured in `pyproject.toml` (line-length 100); `notebooks/` is excluded from `ruff check`/`ruff format` runs (the editor still formats notebooks on save).

## Quick reference

```bash
uv sync                                                            # setup
uv run python scripts/preprocess_antiuav_script.py --step 10       # videos -> frames (resumable)
uv run python scripts/index_antiuav_script.py                      # frames + labels -> index.db
uv run python scripts/datasette_script.py                          # browse the index at http://127.0.0.1:8001
uv run python scripts/dataloader_antiuav_script.py --split val --batch-size 8 --max-batches 5
uv run python scripts/dataloader_antiuav_script.py --split test --conditions FM --drone-only  # per-condition view
uv run python scripts/extract_antiuav_script.py                    # feature extraction over a split
uv run marimo edit notebooks/check_alignment_interactive_antiuav.py  # marimo alignment app
uv run ruff check . && uv run ruff format --check .                # lint/format
```

## Dataset: Anti-UAV RGB-T

The dataset is **not included** in this repo (`/data` is gitignored). Place it at `data/Anti-UAV-RGBT/`:

- `<split>/<sequence>/` contains `visible.mp4` + `visible.json` and `infrared.mp4` + `infrared.json`.
  - Videos: 1000 frames @ 20 fps; RGB 1920x1080, IR 640x512.
  - Per-sequence JSON: `exist` (per-frame drone-presence flags) and `gt_rect` (per-frame `[x, y, w, h]` boxes).
- `label_new/<split>.json` maps each sequence to its adverse-condition tags. The 10 tags (one-hot encoded as columns in the index): `FM, LI, LR, OC, OV, SV, TC, TC-EASY, TC-MID, TC-HARD`.

Local copy: 318 sequences (train 160 / val 67 / test 91); after preprocessing with `--step 10`, 29,727 frames per modality.

### 1. Preprocess: video -> frames

```bash
uv run python scripts/preprocess_antiuav_script.py --step 10
```

Cuts every sequence into `processed/<split>/<seq>/<modality>/<seq>_<modality>_frame_XXXXXX.png`.
Resumable: sequence/modality pairs whose frames already exist are skipped. `--shift N` time-shifts the extracted windows by N frames (RGB and IR stay in sync because the shift is defined once per sequence).

### 2. Index into SQLite

```bash
uv run python scripts/index_antiuav_script.py
```

Builds `data/Anti-UAV-RGBT/index.db` with three tables:

| table            | rows    | columns                                                                 |
|------------------|---------|-------------------------------------------------------------------------|
| `frames_rgb`     | 29,727  | path, sequence, split, frame_id, gt_x/y/w/h, drone_exists, 10 one-hot condition columns |
| `frames_thermal` | 29,727  | same as `frames_rgb`                                                     |
| `sequences`      | 318     | sequence, split, path, num_frames                                        |

### 3. Browse in datasette

```bash
uv run python scripts/datasette_script.py
# then open http://127.0.0.1:8001
```

Serves `index.db` (immutable) with the dataset folder mounted at `/dataset/...`, so frame thumbnails render inline via datasette-render-image-tags. Canned queries live in `data/Anti-UAV-RGBT/datasette-metadata.yml` and can be extended per run with `--query NAME SQL`.

### 4. DataLoader over the index (PyTorch)

```bash
uv run python scripts/dataloader_antiuav_script.py --split val --batch-size 8 --max-batches 5
```

`AntiUAVFrames` (`src/thesis/dataset_antiuav.py`) joins `frames_rgb` and `frames_thermal` on `(sequence, frame_id)` (both modalities are cut up with the same parameters, so they align 1:1) and returns one dict sample per RGB-T pair:

| key            | content                                                              |
|----------------|----------------------------------------------------------------------|
| `rgb`          | transformed RGB image tensor (default: 3x224x224)                   |
| `thermal`      | transformed thermal tensor (RGB by default, or 1-channel grayscale via `thermal_channels=1`) |
| `box_rgb`      | `[4]` float32 GT box (x, y, w, h) in RGB pixel coordinates           |
| `box_thermal`  | `[4]` float32 GT box in thermal pixel coordinates (640x512 space)    |
| `drone_exists` | scalar long                                                          |
| `conditions`   | `[10]` float32 one-hot over `condition_columns`                      |
| `sequence`, `frame_id`, `rgb_path`, `thermal_path` | metadata (strings survive the default collate as lists) |

Options: `split`, `dataset_root`, `db_name`, `transform`/`thermal_transform`, `thermal_channels`, `drone_only`, `condition_columns`, `frac` + `seed` (subsampling). The full index slice stays available as `dataset.df`.

#### Views and transforms

`AntiUAVFrames` subclasses the generic `DataFrameDataset` (`src/thesis/utils/dataset.py`), which provides chainable sub-dataset views over the same df without re-reading `index.db`:

```python
fm = ds.view(conditions=["FM"], how="any", drone_only=True)  # one adverse condition, drones only
both = ds.view(conditions=["LI", "FM"], how="all")  # sequences tagged with both
clean = ds.view(conditions=["neutral"])  # no tags at all (in-distribution; virtual tag NEUTRAL)
ood = ds.view(sequences=["01739", "01740"])  # hand-picked sequences
small = ds.view(frac=0.1, seed=7)  # subsample
train_v = fm.with_transform(train_transform)  # swap transforms on a copy
```

`view(...)` narrows, never widens: a view of a view only shrinks further. Build a fresh `AntiUAVFrames` to widen. `conditions` + `how="any"|"all"` keeps frames whose sequence carries any/all of the given tags; the virtual tag `"neutral"` (`NEUTRAL` in `dataset_antiuav.py`) selects frames whose sequence has no tags at all (none of the condition columns = 1) and combines with real tags like any other condition; `sequences=[...]` restricts to the listed sequences; plus `drone_only` and `frac` + `seed`. `with_transform(transform, thermal_transform=None)` returns a copy that shares the df, so a train view can use augmentation while an eval view of the same frames uses the plain resize. CLI equivalents on `scripts/dataloader_antiuav_script.py`: `--conditions FM LI --condition-how any|all --sequences ... --drone-only --frac --seed` (e.g. `--conditions neutral` for the in-distribution view).

### 5. Notebooks

- `check_antiuav.ipynb` - quick data sanity check (Jupyter).
- `check_alignment_interactive_antiuav.py` / `check_alignment_per_sequence_antiuav.py` - marimo apps for the RGB/IR alignment analysis:

  ```bash
  uv run marimo edit notebooks/check_alignment_interactive_antiuav.py
  ```

- `predict_thermal_edtc_antiuav.ipynb` - first thermal-frame predictions with `model_weights/edtc_yolov5s.pt`.

## Configuration

The `Config` dataclass in `src/thesis/config.py` holds run settings: data dir/name/split/frac/seed/batch size and the model (`yolov5s` with pretrained weights, or `unet`). `config/default.json` mirrors these but is **not loaded by any code yet**; wiring it up (or dropping it) is the first item in `TODO.md`. The dataloader path does not use `Config`: it takes its kwargs directly on `AntiUAVFrames`.

## Notes / known quirks

- The template's toy-dataset code was removed/rewired: `src/thesis/data.py` (targeted a nonexistent `data/boxes` index) is deleted; `src/thesis/extract_antiuav.py` now runs over `AntiUAVFrames`. The Anti-UAV data flow is `preprocess_antiuav.py` -> `index_antiuav.py` -> `index.db` -> `dataset_antiuav.py`.
- `src/thesis/eval.py` and `metrics.py` are placeholders for the upcoming evaluation; `experiments/` and `logs/` are reserved for run outputs (contents gitignored).
- No console script is installed: run the CLIs directly, e.g. `uv run python scripts/dataloader_antiuav_script.py --split val`. (Distribution name: `ua-mm-fusion-uav-detection`; import package: `thesis`.)
- `data/`, `outputs/`, `model_weights/`, `experiments/*`, `logs/*`, `.venv/`, `uv.lock` and caches are gitignored; only code, config, docs and notebooks are tracked.

## Branches

- `torch-dataloader` (checked out) - dataloader + model weights + EDTC prediction notebook + repo restructure
- `main` - preprocessing + indexing + alignment exploration (PR #1 merged)
- `alignment-exploration` - the alignment-analysis branch (merged into main)

## License

MIT, see [LICENSE](LICENSE).
