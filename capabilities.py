"""Capability profiles and runtime policies — the single source of truth.

Two orthogonal concepts govern every execution:

Capabilities  — what the engine CAN do (exposed tools, prompt instructions)
RuntimePolicy — HOW the runtime behaves (budgets, limits, verification depth)

Every endpoint calls build_system_prompt(caps, policy, ...) and receives
a prompt + tool list derived from the preset pair. No mode-specific logic
lives anywhere else.

Presets
-------
CHAT_CAPS / CHAT_POLICY         — web only, light budget, fast iterations
WORKSPACE_CAPS / WORKSPACE_POLICY — read+graph+web, broad context window
AGENT_CAPS / AGENT_POLICY       — full write access, tight working set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ── Capabilities ──────────────────────────────────────────────────────────────

@dataclass
class Capabilities:
    """What the engine is allowed to do. Drives tool exposure and prompt tone."""
    web_search: bool = False
    browse: bool = False

    project_graph: bool = False
    project_memory: bool = False
    read_files: bool = False

    edit_files: bool = False
    write_files: bool = False

    shell: bool = False
    git: bool = False
    ssh: bool = False

    transactions: bool = False


CHAT_CAPS = Capabilities(
    web_search=True,
    browse=True,
)

WORKSPACE_CAPS = Capabilities(
    web_search=True,
    browse=True,
    project_graph=True,
    project_memory=True,
    read_files=True,
)

AGENT_CAPS = Capabilities(
    web_search=True,
    browse=True,
    project_graph=True,
    project_memory=True,
    read_files=True,
    edit_files=True,
    write_files=True,
    shell=True,
    git=True,
    ssh=True,
    transactions=True,
)


# ── Runtime policy ────────────────────────────────────────────────────────────

@dataclass
class RuntimePolicy:
    """How the runtime behaves. Orthogonal to capabilities."""
    max_tool_calls: int = 12
    max_context_tokens: int = 2500
    retrieval_budget: int = 1500
    verification_level: Literal["none", "syntax", "graph", "full"] = "none"
    planning_depth: int = 3
    allow_parallel_tools: bool = False


CHAT_POLICY = RuntimePolicy(
    max_tool_calls=8,
    max_context_tokens=2500,
    retrieval_budget=1500,
    verification_level="none",
    planning_depth=3,
)

WORKSPACE_POLICY = RuntimePolicy(
    max_tool_calls=12,
    max_context_tokens=3000,
    retrieval_budget=2500,
    verification_level="none",
    planning_depth=5,
)

# Agent gets a tighter context window — fast working set beats broad exploration.
AGENT_POLICY = RuntimePolicy(
    max_tool_calls=30,
    max_context_tokens=2000,
    retrieval_budget=1500,
    verification_level="full",
    planning_depth=8,
)


# ── Retrieval policy ──────────────────────────────────────────────────────────

@dataclass
class RetrievalPolicy:
    """How the graph retrieval layer behaves for a given task class."""
    level: Literal["orientation", "edit", "impact", "debug"] = "orientation"
    max_results: int = 12
    expand_depth: int = 2
    prefer_structural: bool = True
    prefer_source: bool = False
    boost_hot_symbols: bool = True
    exclude_paths: list[str] = field(default_factory=list)


# Intent → retrieval policy mapping (selected automatically by task classification)
RETRIEVAL_POLICIES: dict[str, RetrievalPolicy] = {
    "orientation": RetrievalPolicy(level="orientation", max_results=12, expand_depth=1, prefer_structural=True),
    "edit":        RetrievalPolicy(level="edit",        max_results=8,  expand_depth=2, prefer_source=True),
    "impact":      RetrievalPolicy(level="impact",      max_results=10, expand_depth=3, prefer_structural=True),
    "debug":       RetrievalPolicy(level="debug",       max_results=15, expand_depth=2, boost_hot_symbols=True),
}


# ── Tool schema registry ──────────────────────────────────────────────────────

_CAP_TOOLS: list[tuple[str, list[str]]] = [
    ("web_search",     ["web_search"]),
    ("browse",         ["browse_page", "http_request"]),
    ("read_files",     ["read_file", "list_dir"]),
    ("edit_files",     ["write_file", "edit_file"]),
    ("transactions",   ["begin_transaction", "commit_transaction", "rollback_transaction"]),
    ("shell",          ["shell"]),
    ("git",            ["git"]),
    ("ssh",            ["ssh"]),
    ("project_graph",  ["search_symbols", "graph_expand", "get_symbol_source"]),
    ("project_memory", ["manage_memory"]),
]


def get_tool_schemas(caps: Capabilities) -> list[dict]:
    """Return Ollama-format tool schemas for the given capability profile.

    Used by chat / workspace mode (native Ollama tool calling).
    Agent mode generates its own tool listing for the custom JSON protocol.
    """
    from tools import TOOLS
    enabled: set[str] = set()
    for field_name, names in _CAP_TOOLS:
        if getattr(caps, field_name, False):
            enabled.update(names)
    return [TOOLS[n]["schema"] for n in enabled if n in TOOLS]


# ── Prompt generation ─────────────────────────────────────────────────────────

def build_system_prompt(
    caps: Capabilities,
    settings: dict,
    memory: dict,
    project: dict | None = None,
    policy: RuntimePolicy | None = None,
    model: str = "",
    query: str = "",
    session_id: str | None = None,
) -> str:
    """Single prompt generator for all modes.

    Behaviour is entirely driven by `caps` and `policy` — no mode checks.
    - edit_files=True   → full agent protocol with tool listing
    - project_graph=True → graph-aware exploration guidance
    - neither           → plain conversational assistant
    """
    if policy is None:
        policy = CHAT_POLICY

    parts: list[str] = []

    # ── Identity ──────────────────────────────────────────────────────────────
    if caps.edit_files:
        parts.append(
            "You are a coding assistant with full workspace access. "
            "You have the same reasoning and knowledge as in read-only mode — "
            "the difference is that here you can also act."
        )
    elif caps.project_graph:
        parts.append(
            "You are a code intelligence assistant. "
            "You can deeply explore codebases, search the web, and read files "
            "to answer questions about code, architecture, and dependencies. "
            "You cannot modify files."
        )
    else:
        parts.append(
            "You are a helpful, well-rounded assistant. "
            "Answer questions clearly and conversationally. "
            "Only write code or markup if the user explicitly asks for it."
        )

    # ── Reasoning protocol ────────────────────────────────────────────────────
    if caps.edit_files:
        parts.append(f"""
REASONING FIRST:
- Answer from the pre-loaded graph context when you can.
- Use search_symbols / get_symbol_source to go deeper on a specific symbol.
- Use edit_file / shell only when you need to make actual changes.

DECISION ORDER:
1. Graph context already answers this? → Answer directly.
2. Need more detail on a symbol? → search_symbols → get_symbol_source.
3. Need to make an edit? → get_symbol_source → edit_file → shell (verify).
4. Need to run a command? → shell.

Never say "I will …" — just answer or act.
You have up to {policy.max_tool_calls} tool calls.""")
    elif caps.project_graph:
        parts.append("""
EXPLORATION APPROACH:
- Start with the knowledge graph (search_symbols, graph_expand) for code questions.
- Use web_search + browse_page for external docs, RFCs, or current information.
- Combine both to give complete, grounded answers.""")

    # ── Shell / OS guidance (agent only) ──────────────────────────────────────
    if caps.shell:
        from config import IS_WINDOWS, OS_NAME, shell_name
        home = str(Path.home())
        if IS_WINDOWS:
            shell_guidance = (
                f"You are running on Windows. The shell tool uses {shell_name()}.\n"
                "Use PowerShell syntax for local commands (Get-ChildItem, Select-String, "
                "$env:USERPROFILE, Remove-Item).\n"
                "For SSH remotes (Linux/macOS), use standard bash syntax."
            )
        else:
            shell_guidance = (
                f"You are running on {OS_NAME}. The shell tool uses {shell_name()}.\n"
                "Use standard bash/sh syntax."
            )
        parts.append(f"{shell_guidance}\nHome directory: {home}")

    # ── Tool listing (agent only — chat/workspace use native Ollama tool calls) ─
    if caps.edit_files:
        from tools import TOOLS
        from mcp_client import mcp_manager
        tool_lines = []
        for name, t in TOOLS.items():
            fn_schema = t["schema"]["function"]
            props = fn_schema["parameters"]["properties"]
            params = ", ".join(f'{k}: {v.get("type","any")}' for k, v in props.items())
            tool_lines.append(f"- {name}({params}): {fn_schema['description']}")
        tool_lines.extend(mcp_manager.get_tool_descriptions())

        m = model.lower()
        hint = (
            "\nPLANNING: On your FIRST response only, you may start with "
            '"Plan: [one sentence]" before the JSON tool call.'
        )
        if "deepseek" in m:
            hint += (
                "\nFORMAT RULE: When using a tool, output ONLY the JSON object — "
                "nothing before or after it (except the optional Plan: prefix on iteration 1)."
            )
        elif any(x in m for x in ("llama", "mistral", "qwen", "gemma", "phi", "devstral", "solar")):
            hint += (
                "\nFORMAT RULE: Tool calls must be a bare JSON object. "
                "Final answers must be plain text — no JSON."
            )

        parts.append(
            f"\nAvailable tools:\n{chr(10).join(tool_lines)}"
            "\n\nTo use a tool, output ONLY this JSON (with optional Plan: prefix on first response):"
            '\n{"tool": "tool_name", "args": {"param": "value"}}'
            "\n\nWhen done, respond in plain text without any JSON."
            "\n\nRules:\n"
            "- Use tools only when the task requires real data — never fabricate output.\n"
            "- Keep final answers concise but complete.\n"
            "- Never wrap your final answer in JSON.\n"
            f"- Avoid slow commands; always add limits (head, -maxdepth, etc.).{hint}"
        )

    # ── Project context ───────────────────────────────────────────────────────
    if project:
        proj_name = project.get("name", "")
        proj_remote = project.get("remote", "")
        proj_root = project.get("root", "")
        loc = f"{proj_remote}:{proj_root}" if proj_remote else proj_root

        if caps.edit_files:
            ssh_note = (
                f'SSH remote — use ssh tool with remote="{proj_remote}" for file ops'
                if proj_remote
                else "Local — use read_file / write_file / edit_file / shell for file ops"
            )
            parts.append(
                f"\nActive project: {proj_name}\nLocation: {loc}\n"
                f"{ssh_note}\n"
                f"Stay within {proj_root} unless explicitly asked otherwise.\n"
                "Always read current file content before editing."
            )
        else:
            parts.append(f"\nActive project: {proj_name}\nLocation: {loc}")

        # Graph context — budget comes from policy
        if caps.project_graph and query:
            try:
                from db import get_db
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT total_nodes FROM project_state WHERE project_id = ?",
                        (proj_name,),
                    ).fetchone()
                if row and row["total_nodes"]:
                    from compiler.context_builder import build_for_query
                    from compiler.intent import infer_intent
                    intent = infer_intent(query)
                    retrieval_policy = RETRIEVAL_POLICIES.get(intent, RETRIEVAL_POLICIES["orientation"])
                    cb = build_for_query(
                        project_id=proj_name,
                        query=query,
                        intent=intent,
                        session_id=session_id,
                        token_budget=policy.retrieval_budget,
                    )
                    if cb.nodes_shown > 0:
                        if caps.edit_files:
                            parts.append(
                                "\nCODE INTELLIGENCE RULES (MANDATORY — compiled graph is available):\n"
                                "USE search_symbols when finding any function, class, method, interface, or type by name.\n"
                                'USE graph_expand(intent="find") to see what a symbol calls or imports.\n'
                                'USE graph_expand(intent="impact") to see what depends on a symbol (before editing).\n'
                                "USE get_symbol_source for the full implementation when you have a node_id.\n"
                                "USE read_file ONLY when search_symbols returned zero results AND you know the exact file path.\n"
                                "PROHIBITED: list_dir to explore, shell grep/find for symbols, read_file as first step.\n"
                                f"\nContext retrieved for this query:\n{cb.text}"
                            )
                        else:
                            parts.append(f"\nKnowledge graph context:\n{cb.text}")
            except Exception:
                pass
        elif caps.read_files and not caps.project_graph and proj_root:
            try:
                from projects import scan_project_structure
                structure = scan_project_structure(project)
                parts.append("Project structure:\n" + structure[:2000])
            except Exception:
                pass

    # ── User / memory ─────────────────────────────────────────────────────────
    user_name = settings.get("user_name", "")
    if user_name:
        parts.append(f"\nUser: {user_name}.")
    if memory.get("facts"):
        facts = "\n".join(f"- {f['content']}" for f in memory["facts"])
        parts.append(f"\nThings you know about the user:\n{facts}")

    extra_key = "system_prompt_agent" if caps.edit_files else "system_prompt_chat"
    extra = settings.get(extra_key, "")
    if extra:
        parts.append(f"\nAdditional instructions:\n{extra}")

    return "\n".join(p for p in parts if p.strip())
