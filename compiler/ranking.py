"""Centralized scoring function for all retrieval results.

All ranking decisions live here — no scoring logic in fts.py, graph.py,
read_model.py, or search.py.

Signal weights
--------------
w_semantic   = 0.35   (embeddings — 0.0 until implemented)
w_fts        = 0.25   (BM25 text match)
w_graph      = 0.20   (proximity from focus nodes)
w_importance = 0.12   (fan_in — how many callers this node has)
w_hot        = 0.08   (recent access frequency in session)

When semantic = 0, the effective weights normalize automatically because
the denominator in the final score is still 1.0 — the semantic slot just
contributes 0. This means FTS and graph signals carry ~65% of the budget.

Intent modifiers
----------------
Different query intents shift the effective balance:
  "find"    — default weights (user looking for a symbol)
  "impact"  — boost graph backward traversal (who calls this?)
  "edit"    — boost importance (high fan_in = risky to change)
  "debug"   — boost recency (hot symbols + recently changed)
  "explain" — boost fts (name/signature match matters most)

Importance normalization
------------------------
fan_in is raw count (0..N). We clamp at max_fan_in=100 and normalize to [0,1].
Any node with fan_in ≥ 100 is treated as equally "core" (not penalized further).

Hot score
---------
Loaded from the hot_symbols table once per query and passed in as a set.
Boolean: 1.0 if present, 0.0 if not. Future: use access_count for graded score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_FAN_IN = 100.0


# Intent weight overrides (multipliers applied to base weights)
_INTENT_MODIFIERS: dict[str, dict[str, float]] = {
    "find":    {},  # default
    "impact":  {"w_graph": 0.35, "w_importance": 0.05},
    "edit":    {"w_importance": 0.25, "w_fts": 0.15},
    "debug":   {"w_hot": 0.20, "w_fts": 0.20, "w_importance": 0.05},
    "explain": {"w_fts": 0.40, "w_semantic": 0.20},
}


@dataclass
class RankInput:
    node_id: str
    fts_score: float = 0.0        # [0, 1] from fts.py
    graph_score: float = 0.0      # [0, 1] from graph.py (exp decay on depth)
    semantic_score: float = 0.0   # [0, 1] from embeddings (0 until implemented)
    fan_in: int = 0               # raw count from nodes table
    is_hot: bool = False          # present in hot_symbols for this project/session


def _effective_weights(intent: str) -> dict[str, float]:
    base = {
        "w_semantic":   0.35,
        "w_fts":        0.25,
        "w_graph":      0.20,
        "w_importance": 0.12,
        "w_hot":        0.08,
    }
    overrides = _INTENT_MODIFIERS.get(intent, {})
    merged = {**base, **overrides}
    # Re-normalize so weights sum to 1.0 (overrides may shift the total)
    total = sum(merged.values())
    return {k: v / total for k, v in merged.items()}


def score(ri: RankInput, intent: str = "find") -> float:
    """Compute a scalar relevance score in [0, 1]."""
    w = _effective_weights(intent)
    importance_score = min(ri.fan_in, MAX_FAN_IN) / MAX_FAN_IN
    hot_score = 1.0 if ri.is_hot else 0.0

    return (
        w["w_semantic"]   * ri.semantic_score
        + w["w_fts"]        * ri.fts_score
        + w["w_graph"]      * ri.graph_score
        + w["w_importance"] * importance_score
        + w["w_hot"]        * hot_score
    )


def rank(inputs: list[RankInput], intent: str = "find") -> list[tuple[float, RankInput]]:
    """Score and sort a list of RankInputs.

    Returns list of (score, rank_input) sorted descending (best first).
    """
    scored = [(score(ri, intent), ri) for ri in inputs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
