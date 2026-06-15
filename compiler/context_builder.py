"""Context builder — turns SearchResults into a token-budgeted context block.

Takes a list of SearchResult (from read_model.query) and produces a
text block suitable for injection into an LLM system prompt.

Token budget
------------
Each node contributes tokens at one of four levels:
  L0  (~20 tok):  kind + name + path + line range
  L1  (~40 tok):  L0 + signature
  L2  (~60 tok):  L1 + one-line summary
  L3  (~300+ tok): L2 + full source

Level selection is driven by score + remaining budget:
  score >= 0.7    → L3 (source)  if budget allows, else L2
  score >= 0.4    → L2 (summary) if budget allows, else L1
  score >= 0.0    → L1 (signature)

The builder fills highest-scoring nodes first and degrades gracefully
when the budget runs out.

Token estimation
----------------
We estimate tokens as len(text) // 4 (GPT-style approximation).
Accurate enough for budget planning without an actual tokenizer.

Output format
-------------
Returns a structured text block that fits naturally into a system prompt:

  ## Code Context (project: <name>, graph v<version>)

  ### [function] getUserById — src/api/users.ts:42-58
  Signature: async function getUserById(id: string): Promise<User>
  ```typescript
  async function getUserById(id: string): Promise<User> {
    ...
  }
  ```

  ### [class] AuthService — src/auth/service.ts:10-95
  Signature: class AuthService implements IAuthService
  Summary: Handles JWT issuance and session management.

  [+3 more symbols not shown — ask to expand]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from compiler.read_model import SearchResult
from db import get_db


# ── Token budget ──────────────────────────────────────────────────────────────

DEFAULT_TOKEN_BUDGET = 4000   # conservative: leaves room for conversation + tools
MIN_BUDGET_PER_NODE = 20      # always show at least L0


def _tok(text: str) -> int:
    return max(1, len(text) // 4)


# ── Node source fetcher ───────────────────────────────────────────────────────

def _fetch_source(node_id: str) -> str | None:
    with get_db() as conn:
        row = conn.execute("SELECT source FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row["source"] if row else None


# ── Level renderers ───────────────────────────────────────────────────────────

def _render_l0(r: SearchResult) -> str:
    return f"### [{r.kind}] {r.name} — {r.path}:{r.start_line}-{r.end_line}"


def _render_l1(r: SearchResult) -> str:
    lines = [_render_l0(r)]
    if r.signature:
        lines.append(f"Signature: {r.signature}")
    return "\n".join(lines)


def _render_l2(r: SearchResult) -> str:
    lines = [_render_l1(r)]
    if r.summary:
        lines.append(f"Summary: {r.summary}")
    return "\n".join(lines)


def _render_l3(r: SearchResult, source: str) -> str:
    lines = [_render_l1(r)]
    if r.summary:
        lines.append(f"Summary: {r.summary}")
    lang = r.language or ""
    lines.append(f"```{lang}")
    lines.append(source.strip())
    lines.append("```")
    return "\n".join(lines)


# ── Context block ─────────────────────────────────────────────────────────────

@dataclass
class ContextBlock:
    text: str           # ready for system prompt injection
    nodes_shown: int
    nodes_omitted: int
    tokens_used: int
    graph_version: int | None


def build(
    results: list[SearchResult],
    project_id: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> ContextBlock:
    """Build a token-budgeted context block from search results.

    Args:
        results:      ranked SearchResults, best first
        project_id:   for header and state lookup
        token_budget: max tokens to spend on context

    Returns:
        ContextBlock with formatted text and metadata.
    """
    if not results:
        return ContextBlock(
            text="",
            nodes_shown=0,
            nodes_omitted=0,
            tokens_used=0,
            graph_version=None,
        )

    # Fetch graph version for header
    graph_version: int | None = None
    with get_db() as conn:
        row = conn.execute(
            "SELECT graph_version FROM project_state WHERE project_id = ?",
            (project_id,)
        ).fetchone()
        if row:
            graph_version = row["graph_version"]

    header = f"## Code Context (project: {project_id}"
    if graph_version is not None:
        header += f", graph v{graph_version}"
    header += ")\n"

    budget = token_budget - _tok(header)
    sections: list[str] = []
    nodes_shown = 0
    nodes_omitted = 0

    for r in results:
        if budget < MIN_BUDGET_PER_NODE:
            nodes_omitted += 1
            continue

        # Decide level based on score and budget
        score = r.score
        section: str

        if score >= 0.7 and budget >= 80:
            # Try L3 (full source)
            source = _fetch_source(r.node_id)
            if source:
                candidate = _render_l3(r, source)
                cost = _tok(candidate)
                if cost <= budget:
                    section = candidate
                    budget -= cost
                    nodes_shown += 1
                    sections.append(section)
                    continue
            # Fall through to L2
            candidate = _render_l2(r)
            cost = _tok(candidate)
            if cost <= budget:
                section = candidate
                budget -= cost
                nodes_shown += 1
                sections.append(section)
                continue

        if score >= 0.4 and budget >= 30:
            candidate = _render_l2(r)
            cost = _tok(candidate)
            if cost <= budget:
                section = candidate
                budget -= cost
                nodes_shown += 1
                sections.append(section)
                continue

        # L1 — always try if any budget left
        if budget >= MIN_BUDGET_PER_NODE:
            candidate = _render_l1(r)
            cost = _tok(candidate)
            if cost <= budget:
                section = candidate
                budget -= cost
                nodes_shown += 1
                sections.append(section)
                continue

        nodes_omitted += 1

    body = "\n\n".join(sections)
    omit_note = ""
    if nodes_omitted:
        omit_note = f"\n\n[+{nodes_omitted} more symbol(s) not shown — search with more specific terms to expand]"

    full_text = header + body + omit_note

    return ContextBlock(
        text=full_text,
        nodes_shown=nodes_shown,
        nodes_omitted=nodes_omitted,
        tokens_used=_tok(full_text),
        graph_version=graph_version,
    )


def build_for_query(
    project_id: str,
    query: str,
    intent: str = "find",
    focus_nodes: list[str] | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    session_id: str | None = None,
) -> ContextBlock:
    """Convenience: search + build in one call.

    This is the function the agent system-prompt builder calls.
    """
    from compiler.search import find_symbols

    results = find_symbols(
        project_id=project_id,
        query_text=query,
        intent=intent,
        focus_nodes=focus_nodes,
        max_results=30,
        session_id=session_id,
    )
    return build(results, project_id, token_budget)
