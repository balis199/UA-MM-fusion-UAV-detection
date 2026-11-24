# TODO

Prioritized. `AGENTS.md` holds the living conventions.

## Next

1. **Sub-dataset views + DataLoader standardisation** (`src/thesis/dataset_antiuav.py`):
   - add a `view(...)` method on `AntiUAVFrames` that returns a lightweight subset over
     the same index (filters: `drone_only`, condition subset, `frac`+`seed`) — for
     per-condition / OOD evaluation splits without re-reading `index.db`;
   - add a `with_transform(...)` copy so the rgb/thermal transforms can be swapped per
     run instead of only at construction;
   - add one shared `make_dataloader(...)` (batch_size, shuffle, num_workers,
     pin_memory on cuda; dict samples already work with the default collate) and use it
     in `scripts/dataloader_antiuav_script.py` and `src/thesis/extract_antiuav.py`
     instead of hand-rolled `DataLoader(...)` calls.

## Later

2. **Config decision** — `config/default.json` vs the `Config` dataclass are not wired
   together (nothing loads the json; `data_name` default disagrees: dataclass `"boxes"`
   vs json `"visible"`). Decide: add `Config.from_json(config_path)` and use the json as
   the default run config, or drop `config/default.json` and construct `Config` from
   kwargs only. Decide before wiring more CLIs to `Config`.

3. Suggested `utils/` additions
   - `utils/boxes.py` — box format conversions (xywh <-> xyxy, box centers), currently
     re-implemented in `transforms.py` and the alignment notebooks.
   - `utils/plotting.py` — RGB/IR overlay + GT/prediction box drawing, duplicated across
     `notebooks/check_antiuav.ipynb`, `check_alignment_antiuav.ipynb`, `predict_thermal_edtc_antiuav.ipynb`.
   - `utils/logging.py` — run logging into `logs/` (the repo currently uses `print` only).

4. Use `experiments/` and `logs/` for run outputs (dirs kept via `.gitkeep`, contents
   gitignored). `src/thesis/utils/git.py` (`check_git_status`) is there to tag runs with
   the commit hash ("dirty-" prefix on uncommitted changes).

5. Remove or actually use the dead helpers in `preprocess_antiuav.py`:
   `Sequence.window()` and `Frame.resize()` (the time-shift idea now lives in `--shift`).
