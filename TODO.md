# TODO

Prioritized. `AGENTS.md` holds the living conventions.

## Next

1. **Shared `make_dataloader(...)`** in its own module (e.g. `src/thesis/dataloader_antiuav.py`):
   one factory (batch_size, shuffle, num_workers, pin_memory on cuda; dict samples
   already work with the default collate), used in `scripts/dataloader_antiuav_script.py`
   and `src/thesis/extract_antiuav.py` instead of hand-rolled `DataLoader(...)` calls.
