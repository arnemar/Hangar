"""Intent inference — shared by chat and agent."""

from __future__ import annotations


def infer_intent(query: str) -> str:
    """Map a natural-language query to a retrieval intent.

    Intents and their retrieval weight profiles:
      orient  — project overview, architecture (boost importance/fan_in)
      explain — how/why questions (boost FTS + semantic)
      debug   — bug/error investigation (boost hot symbols + FTS)
      edit    — making changes (boost importance = risky-to-change nodes)
      impact  — dependency analysis (boost graph traversal)
      find    — default symbol lookup
    """
    q = query.lower()
    if any(w in q for w in ["about", "overview", "what is this", "what does this", "what's this",
                              "architecture", "structure", "project do", "codebase", "how is it organized"]):
        return "orient"
    if any(w in q for w in ["explain", "how does", "how do", "why does", "walk me through",
                              "what does", "describe", "how is", "how are"]):
        return "explain"
    if any(w in q for w in ["bug", "error", "fail", "broken", "crash", "exception", "wrong",
                              "fix", "issue", "problem", "not working"]):
        return "debug"
    if any(w in q for w in ["change", "modify", "update", "refactor", "rename", "move",
                              "delete", "remove", "add", "implement", "write", "create"]):
        return "edit"
    if any(w in q for w in ["impact", "affect", "depend", "which files", "what uses",
                              "what calls", "what imports", "who calls"]):
        return "impact"
    return "find"
