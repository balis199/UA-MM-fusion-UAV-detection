"""Serve the Anti-UAV RGB-T index database in Datasette, with inline frame thumbnails.

Expects the sqlite index to exist already (build it with `python src/thesis/index_antiuav.py`),
then runs:

    datasette serve --immutable <db> --static dataset:<dataset_root> -m <metadata>

Canned queries are kept in <db-dir>/datasette-metadata.yml, so they persist between
runs and can also be edited by hand. Add new ones per run with --query NAME SQL
(repeatable), e.g. rgb thumbnails:

    --query rgb_thumbs "select 'http://{host}:{port}/dataset/' || replace(path, '\\', '/') as image, * from [frames_rgb] limit 100"

{host} and {port} are substituted. Frames are mounted at /dataset/<path>, so image
URLs render inline via datasette-render-image-tags. Open http://127.0.0.1:8001 to browse.

Defaults live in the configuration block below (DATASET_ROOT, DB_PATH, HOST,
PORT); db/host/port can also be overridden per run with CLI flags.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from thesis.paths import ANTI_UAV_DATA_DIR

# --- Configuration -----------------------------------------------------------
# Change these to serve another dataset/database, or override per run with
# --dataset-root / --db / --host / --port.
DATASET_ROOT = ANTI_UAV_DATA_DIR
DB_PATH = ANTI_UAV_DATA_DIR / "index.db"
HOST = "127.0.0.1"
PORT = 8001

METADATA_NAME = "datasette-metadata.yml"


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    return yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}


def add_query(
    metadata: dict, database_name: str, name: str, sql: str, host: str, port: int
) -> None:
    """Add (or replace) a canned query under databases.<db>.queries.<name>."""
    sql = sql.replace("{host}", host).replace("{port}", str(port))
    queries = (
        metadata.setdefault("databases", {}).setdefault(database_name, {}).setdefault("queries", {})
    )
    queries[name] = {"title": name.replace("_", " "), "sql": sql}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset folder mounted at /dataset/ (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Index database to serve (default: %(default)s)",
    )
    parser.add_argument("--host", default=HOST, help="Host to bind (default: %(default)s)")
    parser.add_argument(
        "--port", type=int, default=PORT, help="Port to bind (default: %(default)s)"
    )
    parser.add_argument(
        "--query",
        nargs=2,
        action="append",
        default=[],
        metavar=("NAME", "SQL"),
        help="Add a canned query NAME with SQL to the metadata file (repeatable)",
    )
    parser.add_argument(
        "--print",
        dest="print_command",
        action="store_true",
        help="Print the datasette command instead of running it",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_root}")
    db_path = args.db.expanduser().resolve()
    if not db_path.exists():
        parser.error(
            f"database missing: {db_path} - build the index first with `python src/thesis/index_antiuav.py`"
        )

    metadata_path = db_path.parent / METADATA_NAME
    metadata = load_metadata(metadata_path)
    changed = not metadata_path.exists()
    for name, sql in args.query:
        add_query(metadata, db_path.stem, name, sql, args.host, args.port)
        print(f"Saved query {name!r} to {metadata_path}")
        changed = True
    if changed:
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    queries = metadata.get("databases", {}).get(db_path.stem, {}).get("queries", {})

    command = [
        sys.executable,
        "-m",
        "datasette",
        "serve",
        "--immutable",
        str(db_path),
        "--static",
        f"dataset:{dataset_root}",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "-m",
        str(metadata_path),
    ]
    print("Running:", " ".join(command))
    print(
        f"Open http://{args.host}:{args.port}/ ({len(queries)} canned queries in {metadata_path})"
    )
    if args.print_command:
        return
    subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
