# TODO

Prioritized.

## Next

1. **Shared `make_dataloader(...)`** in its own module (e.g. `src/thesis/dataloader.py`):
   one factory (batch_size, shuffle, num_workers, pin_memory on cuda; dict samples
   already work with the default collate). Use it in all new code; do not retrofit
   existing scripts (`scripts/antiuav/manual_dataloader.py`, `src/thesis/antiuav/extract.py`
   keep their hand-rolled `DataLoader(...)` calls).

## Later

2. **Config decision**: `config/default.json` vs the `Config` dataclass (`src/thesis/config.py`)
   are not wired together (nothing loads the json; `data_name` default disagrees:
   dataclass `"boxes"` vs json `"visible"`). Decide: add `Config.from_json(config_path)`
   and use the json as the default run config, or drop `config/default.json` and
   construct `Config` from kwargs only. Decide before wiring more CLIs to `Config`.

3. Suggested `src/thesis/utils/` additions
   - `boxes.py`: box format conversions (xywh <-> xyxy, box centers), currently
     re-implemented in `transforms.py` and the alignment notebooks.
   - `plotting.py`: RGB/IR overlay + GT/prediction box drawing, duplicated across
     `notebooks/antiuav/check.ipynb`, `notebooks/antiuav/check_alignment.ipynb`,
     `notebooks/antiuav/predict_thermal_edtc.ipynb`.
   - `logging.py`: run logging into `logs/` (the repo currently uses `print` only).

4. Use `experiments/` and `logs/` for run outputs (dirs kept via `.gitkeep`, contents
   gitignored). `src/thesis/utils/git.py` (`check_git_status`) is there to tag runs with
   the commit hash ("dirty-" prefix on uncommitted changes).

5. Remove or actually use the dead helpers in `src/thesis/antiuav/preprocess.py`:
   `Sequence.window()` and `Frame.resize()` (the time-shift idea now lives in `--shift`).
