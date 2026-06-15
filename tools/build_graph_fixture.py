#!/usr/bin/env python3
"""Build a frozen graph fixture for CI retrieval tests.

Compiles the Hangar project against itself and saves the result to
tests/fixtures/hangar_graph.db. The fixture is then used by
tests/test_retrieval.py without touching the live data/workspace.db.

Usage (from project root):
    python tools/build_graph_fixture.py [--project-root PATH]

The script is idempotent — re-running overwrites the existing fixture.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap sys.path before any project imports ────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Patch db.DB_PATH BEFORE importing any project module ────────────────────
import db  # noqa: E402

_FIXTURE_DIR = _PROJECT_ROOT / "tests" / "fixtures"
_FIXTURE_DB = _FIXTURE_DIR / "hangar_graph.db"
_META_FILE = _FIXTURE_DIR / "hangar_graph_meta.json"

PROJECT_ID = "hangar"


def _get_git_commit(root: Path) -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_fixture(project_root: Path) -> None:
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Use a temp file for the build so a failed compile doesn't corrupt the fixture
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir=_FIXTURE_DIR) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Redirect all DB calls to the temp file
        db.DB_PATH = tmp_path
        db.init_db()

        print(f"[fixture] compiling {project_root} → {tmp_path.name}")

        from compiler.pipeline import compile_project

        result = compile_project(PROJECT_ID, project_root)

        print(
            f"[fixture] compile done — "
            f"files={result.files_compiled} "
            f"nodes=+{result.nodes_created} "
            f"edges=+{result.edges_created} "
            f"errors={result.files_failed}"
        )

        if result.files_failed and result.files_compiled == 0:
            raise RuntimeError(
                f"Compilation failed with {result.files_failed} errors and 0 compiled files.\n"
                f"Errors: {result.errors[:5]}"
            )

        # Verify the fixture is usable
        import sqlite3

        conn = sqlite3.connect(str(tmp_path))
        node_count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE project_id = ?", (PROJECT_ID,)
        ).fetchone()[0]
        fts_check = conn.execute(
            "SELECT COUNT(*) FROM nodes_fts WHERE nodes_fts MATCH 'find*'"
        ).fetchone()[0]
        conn.close()

        print(f"[fixture] verified: {node_count} nodes, FTS match count={fts_check}")

        if node_count < 10:
            raise RuntimeError(
                f"Only {node_count} nodes indexed — something went wrong with extraction."
            )

        # Atomically replace the fixture
        shutil.move(str(tmp_path), str(_FIXTURE_DB))

        commit = _get_git_commit(project_root)
        meta = {
            "project_id": PROJECT_ID,
            "commit": commit,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "node_count": node_count,
            "files_compiled": result.files_compiled,
            "files_failed": result.files_failed,
        }
        _META_FILE.write_text(json.dumps(meta, indent=2))

        print(f"[fixture] saved to {_FIXTURE_DB.relative_to(_PROJECT_ROOT)}")
        print(f"[fixture] metadata: {_META_FILE.relative_to(_PROJECT_ROOT)}")

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
        help="Project root to compile (default: this repo)",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        print(f"error: project root not found: {project_root}", file=sys.stderr)
        sys.exit(1)

    try:
        build_fixture(project_root)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
