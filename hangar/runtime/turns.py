"""Run one chat turn and emit structured events for the browser."""

from contextlib import aclosing
from dataclasses import dataclass
from typing import AsyncIterator

from hangar.integrations.ollama import OllamaError, model_supports_tools, stream_ollama
from hangar.persistence.conversations import messages_for, record_message
from hangar.persistence.storage import load_settings
from hangar.runtime.context import fit_messages, positive_int
from hangar.runtime.options import generation_options
from hangar.runtime.prompt import system_prompt, with_attachments
from hangar.runtime.search import fetch_evidence
from hangar.runtime.tool_round import tool_events
from hangar.workspace.files import LocalProject, WorkspaceError
from hangar.workspace.tools import schemas

MAX_TOOL_ROUNDS = 8
MAX_CALLS_PER_ROUND = 32


@dataclass(frozen=True)
class Turn:
    session_id: str
    model: str
    message: str
    attachments: dict[str, str]
    project: LocalProject | None
    propose_edits: bool
    search_web: bool
    project_issue: str = ""


async def stream_turn(turn: Turn) -> AsyncIterator[dict]:
    settings = load_settings()
    address = settings["ollama_url"]
    context = max(1024, positive_int(settings.get("context_length"), 8192))
    generation = generation_options(settings)
    history = [
        {"role": row["role"], "content": row["content"]}
        for row in messages_for(turn.session_id)
        if row["role"] in {"user", "assistant"} and row["content"]
    ]
    record_message(turn.session_id, "user", turn.message, list(turn.attachments))
    yield {"type": "started"}
    if turn.project_issue:
        yield {"type": "notice", "content": turn.project_issue}

    content = with_attachments(turn.message, turn.attachments, context)
    if turn.search_web:
        yield {"type": "activity", "content": "Searching the web"}
        evidence = await fetch_evidence(turn.message)
        if not evidence:
            yield {"type": "notice", "content": "Web search returned no sources."}
        content += "\n\n" + (evidence or "Web search provided no sources. Say what could not be verified.")

    tools = []
    if turn.project:
        try:
            supported = await model_supports_tools(turn.model, address)
        except OllamaError:
            supported = False
        if supported:
            tools = schemas(turn.propose_edits)
        else:
            yield {"type": "notice", "content": "Project tools are unavailable. Attach files from the project panel."}

    system = system_prompt(settings, turn.project, tools)
    user = {"role": "user", "content": content}
    rounds = []
    for attempt in range(MAX_TOOL_ROUNDS + 1):
        available = tools if attempt < MAX_TOOL_ROUNDS else []
        messages = fit_messages(system, history, user, rounds, context, available)
        if rounds and not any(message["role"] == "tool" for message in messages):
            raise WorkspaceError("The latest tool result could not fit. Increase the context window and try again.")

        answer, thinking, calls = [], [], []
        reason = None
        yield {"type": "activity", "content": "Waiting for Ollama"}
        async with aclosing(stream_ollama(messages, turn.model, available, context, address, generation)) as frames:
            async for frame in frames:
                message = frame.get("message", {})
                if chunk := message.get("content"):
                    answer.append(chunk)
                    yield {"type": "chunk", "content": chunk}
                if chunk := message.get("thinking"):
                    thinking.append(chunk)
                calls.extend(message.get("tool_calls", []))
                if len(calls) > MAX_CALLS_PER_ROUND:
                    raise WorkspaceError("Ollama requested too many project tools at once.")
                if frame.get("done"):
                    reason = frame.get("done_reason")

        text = "".join(answer)
        if calls and available and turn.project:
            if text:
                yield {"type": "reset"}
            reply = {"role": "assistant", "content": text, "tool_calls": calls}
            if thinking:
                reply["thinking"] = "".join(thinking)
            results = [reply]
            for event, result in tool_events(calls, turn.session_id, turn.project, turn.propose_edits):
                if event:
                    yield event
                if result:
                    results.append(result)
            rounds.append(results)
        elif text.strip():
            record_message(turn.session_id, "assistant", text)
            yield {"type": "done", "reason": reason}
            return
        else:
            raise WorkspaceError("Ollama completed the request without a text answer.")
