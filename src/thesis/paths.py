"""Single source of truth for repo paths, plus a helper for dataset-relative paths.

All directory constants are anchored at REPO_ROOT, so they resolve correctly
regardless of the working directory the code (notebook, script, ...) is launched from.

The sqlite index stores frame paths relative to the dataset root, with Windows
separators if the index was built on Windows. `join()` accepts both separator
styles on any host OS, so callers never hand-roll `.replace("\\", "/")`.
"""

from pathlib import Path, PureWindowsPath

# repo root (UA-MM-fusion-UAV-detection), two levels above this file
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
CONFIG_DIR = REPO_ROOT / "config"
MODEL_WEIGHTS_DIR = REPO_ROOT / "model_weights"
ANTI_UAV_DATA_DIR = DATA_DIR / "Anti-UAV-RGBT"


def join(root: Path | str, relative: str | Path) -> Path:
    """Join a dataset-relative path onto ``root``, accepting Windows separators.

    PureWindowsPath parses both ``\\`` and ``/`` as separators on any host OS.
    """
    return Path(root).joinpath(*PureWindowsPath(str(relative)).parts)
