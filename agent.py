"""Agent loop: cancellation, parsing, context management, streaming execution.

Reliability improvements
------------------------
- Retry on malformed tool call: if parse fails but the model clearly tried to
  output a JSON tool call, send a format-correction message and retry once.
- Better final-answer detection: don't treat a mangled tool-call attempt as a
  final answer without first attempting a correction.
- Context compression: when token usage exceeds 70 % of num_ctx, the middle of
  the conversation is summarised so the agent doesn't run out of context on
  long tasks.
- Smart truncation: long tool results are kept as head + tail rather than
  blindly cut at N chars, so critical output at the end isn't lost.
- Tool budget: the model is told upfront how many tool calls it has so it
  doesn't waste them on redundant steps.
- Model-specific hints: deepseek / llama / mistral / qwen each get a short
  formatting reminder that matches their tendencies.
"""

import json
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from config import IS_WINDOWS, OS_NAME, shell_name
from db import get_db
from mcp_client import mcp_manager
from ollama_client import chat_ollama
from storage import get_project, load_memory, load_settings
from tools import TOOLS, execute_tool

# ── Cancellation ──────────────────────────────────────────────────────────────

_active: dict[str, threading.Event] = {}


def register_request(request_id: str) -> threading.Event:
    ev = threading.Event()
    _active[request_id] = ev
    return ev


def cancel_request(request_id: str) -> None:
    if request_id in _active:
        _active[request_id].set()


def finish_request(request_id: str) -> None:
    _active.pop(request_id, None)


def is_cancelled(request_id: str) -> bool:
    ev = _active.get(request_id)
    return ev.is_set() if ev else False


# ── DeepSeek token stripping ──────────────────────────────────────────────────

_DS_PATTERNS = [
    re.compile(r'<｜tool[▁_]calls[▁_]begin｜>[\s\S]*?<｜tool[▁_]calls[▁_]end｜>', re.IGNORECASE),
    re.compile(r'<｜tool[▁_]call[▁_]begin｜>.*?<｜tool[▁_]call[▁_]end｜>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<｜tool[▁_]outputs[▁_]begin｜>[\s\S]*?<｜tool[▁_]outputs[▁_]end｜>', re.IGNORECASE),
    re.compile(r'<｜[^｜]+｜>'),
]


def _strip_ds(text: str) -> str:
    for pat in _DS_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ── Tool-call parsing ─────────────────────────────────────────────────────────

def parse_tool_call(text: str) -> Optional[dict]:
    text = _strip_ds(text).strip()
    if not text:
        return None

    def _valid(d) -> bool:
        return isinstance(d, dict) and "tool" in d and "args" in d

    # 1. Direct parse
    try:
        d = json.loads(text)
        if _valid(d):
            return d
    except Exception:
        pass

    # 2. Inline JSON with tool+args keys
    m = re.search(r'\{[^{}]*"tool"[^{}]*"args"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            if _valid(d):
                return d
        except Exception:
            pass

    # 3. Nested JSON
    for m in re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', text):
        try:
            d = json.loads(m)
            if _valid(d):
                return d
        except Exception:
            pass

    # 4. Fenced code block
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if m:
        try:
            d = json.loads(m.group(1))
            if _valid(d):
                return d
        except Exception:
            pass

    return None


def _looks_like_tool_attempt(text: str) -> bool:
    """Return True if the model clearly tried to write a tool call but formatted it wrong."""
    s = text.strip()
    has_braces = '{' in s and '}' in s
    has_tool_key = '"tool"' in s or "'tool'" in s
    has_args_key = '"args"' in s or "'args'" in s
    short_or_starts_json = len(s) < 600 or s.startswith('{') or s.startswith('```')
    return has_braces and (has_tool_key or has_args_key) and short_or_starts_json


# ── Smart truncation ──────────────────────────────────────────────────────────

def _smart_truncate(text: str, max_chars: int = 2000) -> str:
    """Keep the head and tail of long output so nothing critical is lost."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 50
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n... [{omitted:,} chars omitted] ...\n\n{text[-tail:]}"


# ── Context compression ───────────────────────────────────────────────────────

async def _compress_context(
    messages: list,
    model: str,
    ollama_url: str,
    num_ctx: int,
) -> list:
    """Summarise the middle of the conversation to free up context space.

    Keeps: system message, summary block, last 4 messages.
    Skips if there aren't enough messages to make it worthwhile.
    """
    if len(messages) < 9:
        return messages

    system = messages[0]
    recent = messages[-4:]
    middle = messages[1:-4]

    if not middle:
        return messages

    combined = "\n\n".join(
        f"[{m['role'].upper()}]: {m['content'][:600]}" for m in middle
    )

    summary_msgs = [
        {
            "role": "system",
            "content": (
                "You are a precise summariser. Condense the following agent conversation "
                "into a single short paragraph. Focus on facts discovered and actions taken, "
                "not on the process. Be concrete."
            ),
        },
        {"role": "user", "content": combined},
    ]

    try:
        resp = await chat_ollama(
            summary_msgs, model,
            num_ctx=min(4096, num_ctx),
            ollama_url=ollama_url,
        )
        summary = resp.get("message", {}).get("content", "").strip()
        if summary:
            return [
                system,
                {
                    "role": "user",
                    "content": f"[Summary of earlier steps in this conversation]\n{summary}",
                },
                {"role": "assistant", "content": "Understood, I have the earlier context."},
                *recent,
            ]
    except Exception:
        pass

    return messages


# ── Model-specific formatting hints ───────────────────────────────────────────

def _model_hints(model: str) -> str:
    name = model.lower()
    if "deepseek" in name:
        return (
            "\nFORMAT RULE: When using a tool, output the JSON object with NO text before "
            "or after it. The response must start with { and end with }. "
            "Never wrap it in markdown. Never add an explanation before the JSON."
        )
    if any(x in name for x in ("llama", "mistral", "qwen", "gemma", "phi")):
        return (
            "\nFORMAT RULE: Tool calls must be a bare JSON object only. "
            "Final answers must be plain text only — no JSON."
        )
    return ""


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(
    settings: dict,
    memory: dict,
    project: Optional[dict],
    max_iter: int,
    model: str,
) -> str:
    user_name = settings.get("user_name", "")
    extra = settings.get("system_prompt_agent", "")
    home = str(Path.home())

    tool_lines = []
    for name, t in TOOLS.items():
        fn_schema = t["schema"]["function"]
        props = fn_schema["parameters"]["properties"]
        params = ", ".join(f'{k}: {v.get("type","any")}' for k, v in props.items())
        tool_lines.append(f"- {name}({params}): {fn_schema['description']}")
    tool_lines.extend(mcp_manager.get_tool_descriptions())

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

    prompt = f"""You are an AI agent. You MUST use tools to accomplish tasks — never describe what you would do, just do it.

CRITICAL RULES:
- CREATE a file → call write_file immediately.
- RUN a command → call shell immediately.
- READ a file → call read_file immediately.
- FIND something → call shell or list_dir immediately.
- Never say "I will …" — just use the tool. Action first, explanation after.

{shell_guidance}
Home directory: {home}
Tool call budget: {max_iter} tool calls per message. Use them efficiently.{_model_hints(model)}

Available tools:
{chr(10).join(tool_lines)}

When you need to use a tool, output ONLY this JSON (nothing else before or after):
{{"tool": "tool_name", "args": {{"param": "value"}}}}

When you have enough information, respond in plain text WITHOUT any JSON.

Rules:
- Use tools only when the task requires real data — never fabricate output.
- After tool results, either call another tool or give your final answer.
- Keep final answers concise.
- Never wrap your final answer in JSON.
- Avoid slow commands (find / without -maxdepth); always add limits.
- When asked to describe a project: list the directory first, then read key files."""

    if project:
        from projects import read_project_file, scan_project_structure

        proj_remote = project.get("remote", "")
        proj_root = project.get("root", "")
        loc = f"{proj_remote}:{proj_root}" if proj_remote else proj_root
        structure = scan_project_structure(project)

        readme = ""
        for rname in ("README.md", "readme.md", "README.txt"):
            content = read_project_file(project, rname)
            if not content.startswith("[error]"):
                readme = content[:2000]
                break

        ssh_note = (
            f'SSH remote — use ssh tool with remote="{proj_remote}" for file ops'
            if proj_remote
            else "Local — use read_file / write_file / edit_file / shell for file ops"
        )
        prompt += f"""

Active project: {project.get("name")}
Location: {loc}
{ssh_note}
Stay within {proj_root} unless explicitly asked to go elsewhere.
Always read current file content before editing.

Project structure:
{structure[:800]}

{("README:\n" + readme) if readme else ""}

Use the above context before making tool calls; only call tools for specifics."""

    if user_name:
        prompt += f"\n\nThe user's name is {user_name}."
    if memory["facts"]:
        facts = "\n".join(f"- {f['content']}" for f in memory["facts"])
        prompt += f"\n\nUser context:\n{facts}"
    if extra:
        prompt += f"\n\nAdditional instructions:\n{extra}"

    return prompt


# ── Agent loop ────────────────────────────────────────────────────────────────

async def run_agent_stream(
    session_id: str,
    message: str,
    model: str,
    project_name: Optional[str],
    request_id: str,
    settings: dict,
) -> AsyncGenerator[str, None]:
    cancel_ev = register_request(request_id)
    memory = load_memory()
    max_iter = settings.get("agent_max_iterations", 5)
    num_ctx = settings.get("context_length", 8192)
    ollama_url = settings.get("ollama_url", "")

    project = get_project(project_name) if project_name else None
    system_prompt = _build_system_prompt(settings, memory, project, max_iter, model)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()

    messages = [{"role": "system", "content": system_prompt}]
    for row in rows:
        if row["role"] in ("user", "assistant"):
            messages.append({"role": row["role"], "content": row["content"]})
    messages.append({"role": "user", "content": message})

    all_tool_calls: list[dict] = []
    all_tool_results: list[str] = []
    final_response = ""
    iteration = 0
    total_tokens = 0
    start = time.time()

    while iteration < max_iter:
        iteration += 1

        if is_cancelled(request_id):
            yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
            break

        # ── Context compression ───────────────────────────────────────────────
        if total_tokens > num_ctx * 0.70 or len(messages) > 24:
            messages = await _compress_context(messages, model, ollama_url, num_ctx)
            yield f"data: {json.dumps({'type': 'status', 'message': 'context compressed'})}\n\n"

        yield f"data: {json.dumps({'type': 'thinking', 'iteration': iteration})}\n\n"

        try:
            resp = await chat_ollama(
                messages, model,
                cancel_event=cancel_ev, num_ctx=num_ctx, ollama_url=ollama_url,
            )
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        if resp.get("cancelled"):
            yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
            finish_request(request_id)
            return

        raw = resp.get("message", {}).get("content", "").strip()
        total_tokens += resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)

        if not raw:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Empty response from model'})}\n\n"
            return

        content = _strip_ds(raw)
        if not content:
            break

        tc = parse_tool_call(content)

        # ── Retry on malformed tool-call attempt ──────────────────────────────
        if tc is None and _looks_like_tool_attempt(content):
            yield f"data: {json.dumps({'type': 'status', 'message': 'retrying malformed tool call'})}\n\n"
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    "Your response was not valid JSON. Output ONLY the JSON object, "
                    "nothing else — no explanation, no markdown:\n"
                    '{"tool": "tool_name", "args": {"param": "value"}}'
                ),
            })
            try:
                retry_resp = await chat_ollama(
                    messages, model,
                    cancel_event=cancel_ev, num_ctx=num_ctx, ollama_url=ollama_url,
                )
                if not retry_resp.get("cancelled"):
                    retry_content = _strip_ds(
                        retry_resp.get("message", {}).get("content", "")
                    ).strip()
                    retry_tc = parse_tool_call(retry_content)
                    if retry_tc:
                        # Retry succeeded — use the corrected call
                        tc = retry_tc
                        content = retry_content
                        # Remove the correction exchange from messages
                        messages.pop()
                        messages.pop()
                    else:
                        # Retry failed too — remove correction exchange, treat as final answer
                        messages.pop()
                        messages.pop()
            except Exception:
                pass

        if tc:
            tool_name = tc.get("tool", "")
            tool_args = tc.get("args", {})

            yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': tool_args})}\n\n"
            if tool_name.startswith("mcp__"):
                result = await mcp_manager.call_tool(tool_name, tool_args)
            else:
                result = await execute_tool(tool_name, tool_args)

            if is_cancelled(request_id):
                yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                finish_request(request_id)
                return

            # Smart truncate: head + tail so tail output isn't lost
            truncated = _smart_truncate(result, max_chars=2000)
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': _smart_truncate(result, 1000)})}\n\n"

            all_tool_calls.append({"tool": tool_name, "args": tool_args})
            all_tool_results.append(result)

            messages.append({"role": "assistant", "content": content})
            remaining = max_iter - iteration

            if iteration >= 2:
                # Force a final answer after 2+ tool calls
                force_msgs = [m for m in messages if m["role"] != "system"]
                force_msgs.insert(0, {
                    "role": "system",
                    "content": (
                        "Answer the user's question directly based on the information gathered. "
                        "Plain text only — no JSON, no tool calls."
                    ),
                })
                force_msgs.append({
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n{truncated}\n\n"
                        f"You have {remaining} tool call(s) remaining. "
                        "If you have enough information, give your final answer now in plain text."
                    ),
                })
                try:
                    forced = await chat_ollama(
                        force_msgs, model,
                        cancel_event=cancel_ev, num_ctx=num_ctx, ollama_url=ollama_url,
                    )
                    if not forced.get("cancelled"):
                        ans = forced.get("message", {}).get("content", "").strip()
                        if ans and not _looks_like_tool_attempt(ans):
                            final_response = ans
                            yield f"data: {json.dumps({'type': 'response', 'content': ans})}\n\n"
                            break
                except Exception:
                    pass
                # Forced answer failed or looked like another tool call — let loop continue
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n{truncated}\n\n"
                        f"You have {remaining} tool call(s) remaining. Answer now."
                    ),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n{truncated}\n\n"
                        f"You have {remaining} tool call(s) remaining. "
                        "Use another tool if needed, otherwise give your final answer in plain text."
                    ),
                })
        else:
            # No tool call — this is the final answer
            final_response = content
            yield f"data: {json.dumps({'type': 'response', 'content': final_response})}\n\n"
            break

    # ── Hit iteration limit ───────────────────────────────────────────────────
    if not final_response and all_tool_results:
        messages.append({
            "role": "user",
            "content": (
                "You have used all your tool calls. "
                "Give your final answer based only on what you have gathered. No more tools."
            ),
        })
        try:
            resp = await chat_ollama(
                messages, model,
                cancel_event=cancel_ev, num_ctx=num_ctx, ollama_url=ollama_url,
            )
            final_response = resp.get("message", {}).get("content", "").strip()
        except Exception:
            pass
        if not final_response:
            final_response = (
                f"Reached tool call limit ({max_iter} steps). Partial results:\n"
                + "\n".join(f"- {r[:200]}" for r in all_tool_results[:3])
            )
        yield f"data: {json.dumps({'type': 'response', 'content': final_response})}\n\n"

    # ── Persist to DB ─────────────────────────────────────────────────────────
    tc_json = json.dumps(all_tool_calls) if all_tool_calls else None
    tr_json = json.dumps(all_tool_results) if all_tool_results else None
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), session_id, "assistant",
             final_response, tc_json, tr_json, now),
        )
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))

    elapsed = round(time.time() - start, 1)
    finish_request(request_id)
    yield f"data: {json.dumps({'type': 'done', 'total_tokens': total_tokens, 'elapsed': elapsed, 'iterations': iteration})}\n\n"
