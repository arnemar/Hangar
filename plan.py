"""Externalized task state — the runtime owns this, the model reads it.

The model never has to remember where it is in a multi-step task.
Instead: model proposes → runtime executes → Plan updates → model sees state.

Two task shapes use the Plan differently:
  Implementation  steps[] + completed_steps[int] = structured workflow
  Debugging       hypotheses[] + evidence[]       = investigative trail

render() stays bounded at ~100-200 tokens — it is injected on every iteration.
Everything else lives in the events table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Working set entry ─────────────────────────────────────────────────────────

@dataclass
class WorkingItem:
    symbol: str
    reason: Literal["retrieved", "edited", "dependency", "hypothesis"]

    def __str__(self) -> str:
        return f"{self.symbol} ({self.reason})"


# ── Plan ──────────────────────────────────────────────────────────────────────

@dataclass
class Plan:
    goal: str

    # Implementation path
    steps: list[str] = field(default_factory=list)
    current_step: int = 0
    completed_steps: list[int] = field(default_factory=list)  # indices into steps

    # Investigative path (debugging / exploration)
    hypotheses: list[str] = field(default_factory=list)   # hard cap: 5
    evidence: list[str] = field(default_factory=list)      # hard cap: 10

    # Shared state
    working_set: list[WorkingItem] = field(default_factory=list)  # hard cap: 10
    transaction_id: str | None = None
    verification_state: Literal["pending", "passed", "failed", "skipped"] = "pending"

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def set_steps(self, text: str) -> None:
        """Parse a plan string (possibly multi-line) into discrete steps."""
        lines = [ln.strip().lstrip("-•*0123456789.). ") for ln in text.splitlines()]
        self.steps = [ln for ln in lines if ln]
        self.current_step = 0
        self.completed_steps = []

    def add_hypothesis(self, text: str) -> None:
        """Add a hypothesis, capped at 5."""
        text = text.strip()
        if text and text not in self.hypotheses:
            if len(self.hypotheses) >= 5:
                self.hypotheses.pop(0)
            self.hypotheses.append(text)

    def add_evidence(self, text: str) -> None:
        """Record a piece of evidence (supports or refutes a hypothesis), capped at 10."""
        text = text.strip()
        if text:
            if len(self.evidence) >= 10:
                self.evidence.pop(0)
            self.evidence.append(text)

    def advance(self, tool_name: str, tool_args: dict) -> None:
        """Record a completed tool call; update step index and working set."""
        # Step tracking (implementation path)
        if self.steps and self.current_step < len(self.steps):
            self.completed_steps.append(self.current_step)
            self.current_step += 1

        # Look up semantic metadata declared on the tool — no hard-coded inference here
        from tools import TOOLS, ToolMetadata
        meta: ToolMetadata = TOOLS.get(tool_name, {}).get("meta", ToolMetadata())

        if not meta.affects_working_set:
            if tool_args.get("transaction_id"):
                self.transaction_id = str(tool_args["transaction_id"])
            return

        reason = meta.working_set_reason
        for key, val in tool_args.items():
            if not isinstance(val, str):
                continue
            if not any(k in key for k in ("path", "file", "symbol", "name", "query")):
                continue
            short = _short_name(val)
            if not short:
                continue
            existing = next((w for w in self.working_set if w.symbol == short), None)
            if existing:
                existing.reason = reason   # promote to latest reason
            else:
                if len(self.working_set) >= 10:
                    self.working_set.pop(0)
                self.working_set.append(WorkingItem(symbol=short, reason=reason))

        if tx := tool_args.get("transaction_id"):
            self.transaction_id = str(tx)

    def set_verification(self, state: Literal["pending", "passed", "failed", "skipped"]) -> None:
        self.verification_state = state

    def is_complete(self) -> bool:
        return bool(self.steps) and self.current_step >= len(self.steps)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Return a compact state block (~100-200 tokens) for model context injection.

        Empty string when no meaningful state exists yet (first iteration).
        """
        has_content = self.steps or self.hypotheses or self.working_set or self.evidence
        if not has_content:
            return ""

        lines = ["[TASK STATE]", f"Goal: {self.goal}"]

        # Implementation path: step progress
        if self.steps:
            for i, step in enumerate(self.steps):
                if i in self.completed_steps:
                    lines.append(f"  ✓ {step}")
                elif i == self.current_step:
                    lines.append(f"  → {step}")
                else:
                    lines.append(f"    {step}")

        # Investigative path: hypotheses
        if self.hypotheses:
            lines.append("Hypotheses:")
            for h in self.hypotheses:          # already capped at 5
                lines.append(f"  ? {h}")

        # Evidence (last 3 to save tokens)
        if self.evidence:
            lines.append("Evidence:")
            for e in self.evidence[-3:]:
                lines.append(f"  • {e}")

        # Working set (last 8)
        if self.working_set:
            ws = ", ".join(str(w) for w in self.working_set[-8:])
            lines.append(f"Working: {ws}")

        if self.transaction_id:
            lines.append(f"Transaction: {self.transaction_id}")

        if self.verification_state != "pending":
            lines.append(f"Verification: {self.verification_state}")

        lines.append("[/TASK STATE]")
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _short_name(val: str) -> str:
    """Extract a readable short name from a path or symbol string."""
    val = val.strip()
    if not val or len(val) > 120:
        return ""
    if "/" in val:
        return val.split("/")[-1]
    if "\\" in val:
        return val.split("\\")[-1]
    return val[:60]
