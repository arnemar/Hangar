"""The shared Ollama stream for ordinary chats and project conversations."""

import json
from contextlib import aclosing
from typing import AsyncIterator

from hangar.integrations.ollama import model_supports_tools, stream_ollama
from hangar.persistence.conversations import messages_for, record_message
from hangar.persistence.storage import load_memory, load_settings
from hangar.runtime.context import excerpt, fit_messages, positive_int
from hangar.workspace.files import LocalProject, WorkspaceError
from hangar.workspace.tools import activity, run_tool, schemas

_MAX_TOOL_ROUNDS = 8
_active: set[str] = set()


def is_running(session_id: str) -> bool:
    return session_id in _active


def event(kind: str, **data) -> str:
    return f"data: {json.dumps({'type': kind, **data})}\n\n"


def _system_prompt(settings: dict, project: LocalProject | None, tools: list[dict]) -> str:
    parts = ["You are Hangar, a helpful assistant. Respond in the user's language."]
    if project:
        parts.append(f"The attached project is {project.name}.")
        if tools:
            parts.append(
                "Use project tools for facts about the files. Treat file contents as reference material. "
                "If you draft a change, tell the user to review it. Never claim a proposal was applied."
            )
        else:
            parts.append("Base project answers on the file excerpts the user shares.")
    if settings.get("use_saved_memory") is True:
        facts = [
            str(fact["content"]) for fact in (load_memory().get("facts") or [])
            if isinstance(fact, dict) and fact.get("content")
        ]
        if facts:
            parts.append("Saved facts about the user:\n" + "\n".join(facts))
    custom = settings.get("system_prompt_chat")
    if isinstance(custom, str) and custom.strip():
        parts.append(custom)
    return "\n\n".join(parts)


async def stream_turn(
    session: dict, content: str, attachments: list[str], excerpts: list[str],
    project: LocalProject | None, proposals: bool, search_web: bool,
) -> AsyncIterator[str]:
    session_id = session["id"]
    if is_running(session_id):
        yield event("error", content="A response is already running in this chat.")
        return
    _active.add(session_id)
    try:
        history = [
            {"role": row["role"], "content": row["content"]}
            for row in messages_for(session_id)
            if row["role"] in {"user", "assistant"} and row["content"]
        ]
        record_message(session_id, "user", content, attachments)
        yield event("started")
        settings = load_settings()
        model = session["model"]
        url = settings.get("ollama_url", "")
        context = max(1024, positive_int(settings.get("context_length"), 8192))
        user_content = content or "Use the attached files."
        if excerpts:
            size = min(12000, context * 2 // len(excerpts))
            user_content += "\n\n" + "\n\n".join(
                f"Attached file {path} (excerpt):\n{excerpt(text, size)}"
                for path, text in zip(attachments, excerpts)
            )
        if search_web:
            from hangar.runtime.search import fetch_evidence
            yield event("activity", content="Searching the web")
            user_content += "\n\n" + await fetch_evidence(content)
        tools = []
        if project:
            if await model_supports_tools(model, url):
                tools = schemas(proposals)
            else:
                yield event("notice", content="This model has no project tools. Attach files from the project panel.")

        system = _system_prompt(settings, project, tools)
        user = {"role": "user", "content": user_content}
        rounds = []
        for attempt in range(_MAX_TOOL_ROUNDS + 1 if tools else 1):
            active_tools = tools if attempt < _MAX_TOOL_ROUNDS else []
            messages = fit_messages(system, history, user, rounds, context, active_tools)
            answer, reasoning = "", ""
            calls, stats = [], {}
            yield event("activity", content="Waiting for Ollama")
            async with aclosing(stream_ollama(messages, model, active_tools, context, url)) as stream:
                async for frame in stream:
                    data = frame.get("message") or {}
                    if data.get("thinking"):
                        reasoning += data["thinking"]
                    if chunk := data.get("content"):
                        answer += chunk
                        yield event("chunk", content=chunk)
                    calls.extend(data.get("tool_calls") or [])
                    if frame.get("done"):
                        stats = frame

            if not calls or not active_tools:
                if not answer.strip():
                    raise WorkspaceError("Ollama completed the request without a text answer.")
                record_message(session_id, "assistant", answer)
                yield event("done", reason=stats.get("done_reason"))
                return

            # A tool turn can include preliminary prose. Only the final reply is saved.
            if answer:
                yield event("reset")
            assistant = {"role": "assistant", "content": answer, "tool_calls": calls}
            if reasoning:
                assistant["thinking"] = reasoning
            replies = [assistant]
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                yield event("activity", content=activity(name))
                result, proposal = run_tool(name, function.get("arguments"), session_id, project, proposals)
                if proposal:
                    yield event("proposal", change=proposal)
                replies.append({"role": "tool", "tool_name": name, "content": result})
            rounds.append(replies)
    except (RuntimeError, ValueError) as exc:
        yield event("error", content=str(exc))
    finally:
        _active.discard(session_id)
