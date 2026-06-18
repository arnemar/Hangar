"""Agent loop: streaming, planning, cancellation, context management.

Key behaviours
--------------
- Streaming: every token from the model is forwarded live as agent_stream
  events.  Tool-call JSON is cleared from the UI when the tool_call event
  arrives; final-answer text stays as the response.
- Planning: if the model starts its first response with "Plan:", the line is
  extracted and sent as a separate 'plan' event so the user sees the model's
  intent before it starts using tools.
- No premature forced answer: the old "force final at iteration 2" logic is
  removed.  The model can use all available iterations freely; we only force
  a final answer when the very last iteration is exhausted.
- Higher default iteration cap (12) so multi-step coding tasks complete.
- Retry on malformed tool call: if parse fails but the model clearly tried to
  output a JSON tool call, send a format-correction message and retry once.
- Context compression: when token usage exceeds 70 % of num_ctx, the middle
  of the conversation is summarised to keep the context window free.
- Smart truncation: long tool results keep head + tail so nothing critical is
  lost at the end.
"""

import json
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from capabilities import AGENT_CAPS, AGENT_POLICY, build_system_prompt
from plan import Plan
from config import IS_WINDOWS, OS_NAME, shell_name
from db import get_db
from mcp_client import mcp_manager
from ollama_client import chat_ollama, stream_ollama
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


_THINK_RE = re.compile(r'<think>[\s\S]*?</think>', re.IGNORECASE)


def _strip_ds(text: str) -> str:
    text = _THINK_RE.sub("", text)
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
    s = text.strip()
    has_braces = '{' in s and '}' in s
    has_tool_key = '"tool"' in s or "'tool'" in s
    has_args_key = '"args"' in s or "'args'" in s
    short_or_starts_json = len(s) < 600 or s.startswith('{') or s.startswith('```')
    return has_braces and (has_tool_key or has_args_key) and short_or_starts_json


# ── Smart truncation ──────────────────────────────────────────────────────────

def _smart_truncate(text: str, max_chars: int = 2000) -> str:
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
                    "content": f"[Summary of earlier steps]\n{summary}",
                },
                {"role": "assistant", "content": "Understood, continuing from where we left off."},
                *recent,
            ]
    except Exception:
        pass

    return messages


# ── Model-specific formatting hints ───────────────────────────────────────────

def _model_hints(model: str) -> str:
    name = model.lower()
    base = (
        "\nPLANNING: On your FIRST response only, you may start with "
        '"Plan: [one sentence]" before the JSON tool call. '
        "This helps you stay on track."
    )
    if "deepseek" in name:
        return (
            base +
            "\nFORMAT RULE: When using a tool, output ONLY the JSON object — "
            "nothing before or after it (except the optional Plan: prefix on iteration 1). "
            "No markdown, no explanation."
        )
    if any(x in name for x in ("llama", "mistral", "qwen", "gemma", "phi", "devstral", "solar")):
        return (
            base +
            "\nFORMAT RULE: Tool calls must be a bare JSON object. "
            "Final answers must be plain text — no JSON."
        )
    return base


# ── System prompt (delegated to capabilities.py) ──────────────────────────────

def _build_system_prompt(
    settings: dict,
    memory: dict,
    project: Optional[dict],
    model: str,
    query: str = "",
    session_id: str | None = None,
) -> str:
    return build_system_prompt(
        caps=AGENT_CAPS,
        policy=AGENT_POLICY,
        settings=settings,
        memory=memory,
        project=project,
        model=model,
        query=query,
        session_id=session_id,
    )


# ── Planner ───────────────────────────────────────────────────────────────────

async def _stream_plan(
    task: str,
    planner_model: str,
    ollama_url: str,
    num_ctx: int,
) -> AsyncGenerator[str, None]:
    """Stream planner output as SSE events (think_stream, think_end, plan_stream, plan)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a task planner for an AI coding agent. "
                "Given a coding task, output a concise numbered plan (3–7 steps). "
                "Each step must be one concrete action: read a specific file, run a "
                "command, make a specific edit, or verify something. "
                "Do NOT write code. Do NOT explain. Output the numbered list only.\n\n"
                "Example:\n"
                "1. Read src/app.py to understand the route structure\n"
                "2. Find the broken handler in routes/users.py\n"
                "3. Fix the off-by-one error in get_page()\n"
                "4. Run pytest tests/test_users.py to verify"
            ),
        },
        {"role": "user", "content": f"Task: {task}"},
    ]
    plan_buffer = ""
    was_thinking = False
    try:
        async for line in stream_ollama(messages, planner_model, num_ctx=min(4096, num_ctx), ollama_url=ollama_url):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = data.get("message", {})
            think_chunk = msg.get("thinking", "")
            chunk = msg.get("content", "")
            if think_chunk:
                was_thinking = True
                yield f"data: {json.dumps({'type': 'think_stream', 'content': think_chunk, 'model': planner_model})}\n\n"
            if chunk:
                if was_thinking:
                    was_thinking = False
                    yield f"data: {json.dumps({'type': 'think_end'})}\n\n"
                plan_buffer += chunk
                yield f"data: {json.dumps({'type': 'plan_stream', 'content': chunk, 'model': planner_model})}\n\n"
            if data.get("done"):
                break
    except Exception:
        pass
    if was_thinking:
        yield f"data: {json.dumps({'type': 'think_end'})}\n\n"
    plan_text = _THINK_RE.sub("", plan_buffer).strip()
    yield f"data: {json.dumps({'type': 'plan', 'content': plan_text, 'model': planner_model})}\n\n"


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
    max_iter = AGENT_POLICY.max_tool_calls
    num_ctx = settings.get("context_length", 8192)
    ollama_url = settings.get("ollama_url", "")

    project = get_project(project_name) if project_name else None
    system_prompt = _build_system_prompt(settings, memory, project, model, query=message, session_id=session_id)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()

    messages = [{"role": "system", "content": system_prompt}]
    for row in rows:
        if row["role"] in ("user", "assistant"):
            messages.append({"role": row["role"], "content": row["content"]})

    # ── Optional planner pre-pass ─────────────────────────────────────────────
    planner_model = settings.get("planner_model", "").strip()
    plan_text = ""
    if planner_model and planner_model != model:
        async for event in _stream_plan(message, planner_model, ollama_url, num_ctx):
            yield event
            try:
                d = json.loads(event.removeprefix("data: ").rstrip("\n"))
                if d.get("type") == "plan":
                    plan_text = d.get("content", "")
            except Exception:
                pass

    # Build user turn — inject plan as context when available
    if plan_text:
        messages.append({
            "role": "user",
            "content": (
                f"{message}\n\n"
                f"Execution plan (follow this order):\n{plan_text}"
            ),
        })
    else:
        messages.append({"role": "user", "content": message})

    all_tool_calls: list[dict] = []
    all_tool_results: list[str] = []
    final_response = ""
    iteration = 0
    total_tokens = 0
    start = time.time()
    plan = Plan(goal=message)

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

        # ── Stream the model response token-by-token ──────────────────────────
        raw = ""
        was_thinking = False
        iter_tokens = 0

        try:
            async for line in stream_ollama(messages, model, num_ctx=num_ctx, ollama_url=ollama_url):
                if cancel_ev.is_set():
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    finish_request(request_id)
                    return
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                msg = data.get("message", {})
                think_chunk = msg.get("thinking", "")
                chunk = msg.get("content", "")
                if think_chunk:
                    was_thinking = True
                    yield f"data: {json.dumps({'type': 'think_stream', 'content': think_chunk, 'model': model})}\n\n"
                if chunk:
                    if was_thinking:
                        was_thinking = False
                        yield f"data: {json.dumps({'type': 'think_end'})}\n\n"
                    raw += chunk
                    yield f"data: {json.dumps({'type': 'agent_stream', 'content': chunk})}\n\n"
                if data.get("done"):
                    iter_tokens = (
                        data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
                    )
                    break
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return

        total_tokens += iter_tokens

        if is_cancelled(request_id):
            yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
            finish_request(request_id)
            return

        if not raw:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Empty response from model'})}\n\n"
            return

        content = _strip_ds(raw)
        if not content:
            break

        # ── Extract optional plan from first response ─────────────────────────
        if iteration == 1 and content.lstrip().startswith("Plan:"):
            content = content.lstrip()
            newline = content.find("\n")
            if newline > 0:
                plan_text = content[5:newline].strip()
                content = content[newline:].strip()
            else:
                plan_text = content[5:].strip()
                content = ""
            if plan_text:
                plan.set_steps(plan_text)
                yield f"data: {json.dumps({'type': 'plan', 'content': plan_text})}\n\n"

        if not content:
            continue

        tc = parse_tool_call(content)

        # ── Retry on malformed tool-call attempt ──────────────────────────────
        if tc is None and _looks_like_tool_attempt(content):
            yield f"data: {json.dumps({'type': 'status', 'message': 'retrying malformed tool call'})}\n\n"
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    "Your response was not valid JSON. Output ONLY the JSON object, "
                    "nothing else:\n"
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
                        tc = retry_tc
                        content = retry_content
                        messages.pop()
                        messages.pop()
                    else:
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

            truncated = _smart_truncate(result, max_chars=2000)
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': _smart_truncate(result, 1000)})}\n\n"

            all_tool_calls.append({"tool": tool_name, "args": tool_args})
            all_tool_results.append(result)
            plan.advance(tool_name, tool_args)
            plan_state = plan.render()
            if plan_state:
                yield f"data: {json.dumps({'type': 'plan_state', 'content': plan_state, 'completed': len(plan.completed_steps), 'total': len(plan.steps) or len(plan.completed_steps)})}\n\n"

            messages.append({"role": "assistant", "content": content})
            remaining = max_iter - iteration

            if remaining == 0:
                # Last iteration: force a final answer from accumulated results
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
                        "You have used all available tool calls. "
                        "Give your complete final answer now in plain text."
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
            else:
                plan_state = plan.render()
                plan_block = f"\n\n{plan_state}" if plan_state else ""
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n{truncated}{plan_block}\n\n"
                        f"You have {remaining} tool call(s) remaining. "
                        "Use another tool if needed, otherwise give your final answer in plain text."
                    ),
                })
        else:
            # No tool call → this is the final answer
            final_response = content
            yield f"data: {json.dumps({'type': 'response', 'content': final_response})}\n\n"
            break

    # ── Hit iteration limit without a final answer ────────────────────────────
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
