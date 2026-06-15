"""Graph CI: retrieval regression tests against a frozen graph fixture.

Run tools/build_graph_fixture.py first to generate tests/fixtures/hangar_graph.db.
Then: pytest tests/test_retrieval.py -v

Each test case asserts that a natural-language query returns at least min_recall
of the expected symbols in the top TOP_K results. Cases whose expected nodes are
absent from the fixture (renamed/deleted symbols) are skipped, not failed.
"""

# Patch db.DB_PATH BEFORE any project imports to redirect all DB calls to fixture
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import db  # noqa: E402 — must patch before other imports
FIXTURE = Path(__file__).parent / "fixtures" / "hangar_graph.db"
db.DB_PATH = FIXTURE

import json  # noqa: E402
import sqlite3  # noqa: E402

import pytest  # noqa: E402

from compiler.search import find_symbols  # noqa: E402

PROJECT_ID = "hangar"
EVAL_FILE = Path(__file__).parent / "eval" / "retrieval_cases.jsonl"
TOP_K = 10


def _load_cases() -> list[dict]:
    if not EVAL_FILE.exists():
        return []
    cases = []
    with open(EVAL_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


_CASES = _load_cases()


@pytest.fixture(scope="session")
def graph_db():
    """Session-scoped fixture — skip all retrieval tests if snapshot is missing."""
    if not FIXTURE.exists():
        pytest.skip(
            f"Graph fixture not found at {FIXTURE.relative_to(_ROOT)}\n"
            "Run:  python tools/build_graph_fixture.py"
        )
    return FIXTURE


def _resolve_expected(expected: list[dict], conn: sqlite3.Connection) -> list[str]:
    """Translate (name, path, kind) tuples to current node IDs."""
    node_ids = []
    for exp in expected:
        row = conn.execute(
            "SELECT id FROM nodes WHERE name = ? AND path = ? AND kind = ? AND project_id = ?",
            (exp["name"], exp["path"], exp["kind"], PROJECT_ID),
        ).fetchone()
        if row:
            node_ids.append(row[0])
    return node_ids


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[c["query"][:50] for c in _CASES] if _CASES else [],
)
def test_retrieval_recall(case: dict, graph_db: Path) -> None:
    conn = sqlite3.connect(str(graph_db))
    try:
        expected_ids = _resolve_expected(case["expected"], conn)
    finally:
        conn.close()

    if not expected_ids:
        pytest.skip(
            f"Stale eval case — expected nodes not in fixture: {case['expected']}\n"
            "Update tests/eval/retrieval_cases.jsonl or rebuild the fixture."
        )

    results = find_symbols(PROJECT_ID, case["query"], max_results=TOP_K)
    result_ids = {r.node_id for r in results}

    min_recall = case.get("min_recall", 1)
    found = sum(1 for eid in expected_ids if eid in result_ids)

    top5 = [(r.name, r.path, round(r.score, 3)) for r in results[:5]]
    assert found >= min_recall, (
        f"\nQuery: {case['query']!r}\n"
        f"Expected {min_recall}/{len(expected_ids)} expected node(s) in top {TOP_K}, found {found}\n"
        f"Expected symbols: {[e['name'] for e in case['expected']]}\n"
        f"Top {min(5, len(results))} results: {top5}"
    )


def test_fixture_has_nodes(graph_db: Path) -> None:
    """Sanity check: fixture must have a reasonable number of symbols."""
    conn = sqlite3.connect(str(graph_db))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE project_id = ?", (PROJECT_ID,)
        ).fetchone()[0]
    finally:
        conn.close()

    assert count >= 50, (
        f"Fixture has only {count} nodes — fixture may be empty or corrupt.\n"
        "Run: python tools/build_graph_fixture.py"
    )


def test_fixture_fts_populated(graph_db: Path) -> None:
    """FTS5 virtual table must be in sync (not empty)."""
    conn = sqlite3.connect(str(graph_db))
    try:
        # A direct FTS5 MATCH that should definitely hit something
        rows = conn.execute(
            "SELECT COUNT(*) FROM nodes_fts WHERE nodes_fts MATCH 'find*'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert rows > 0, "FTS5 index appears empty — triggers may not have fired during fixture build"
