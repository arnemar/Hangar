"""Fit recent conversation and tool rounds into the requested context window."""

import json


def positive_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed > 0 else default


def _estimate(messages: list[dict]) -> int:
    return sum(
        8 + (len(json.dumps(message, ensure_ascii=False).encode("utf-8")) + 3) // 4
        for message in messages
    )


def excerpt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n\n[... {len(text) - max_chars:,} characters omitted ...]\n\n{text[-tail:]}"


def _history_turns(history: list[dict]) -> list[list[dict]]:
    turns: list[list[dict]] = []
    for message in history:
        if message["role"] == "user" or not turns:
            turns.append([])
        turns[-1].append(message)
    return turns


def fit_messages(
    system_prompt: str,
    history: list[dict],
    user_turn: dict,
    tool_rounds: list[list[dict]],
    num_ctx: int,
    tools: list[dict] | None = None,
    final_instruction: dict | None = None,
) -> list[dict]:
    """Keep the current request intact and fill remaining space with recent turns."""
    system = [{"role": "system", "content": system_prompt}]
    suffix = [final_instruction] if final_instruction else []
    # Leave room for generation and template overhead. The estimate is approximate.
    remaining = num_ctx * 4 // 5 - _estimate(system + [user_turn] + suffix + (tools or []))
    recent_rounds = []
    for round_messages in reversed(tool_rounds):
        cost = _estimate(round_messages)
        if cost > remaining:
            if not recent_rounds and remaining > 0:
                # Preserve the latest call and reply linkage when its result is large.
                result_positions = [
                    i for i, message in enumerate(round_messages)
                    if message["role"] == "tool" or i > 0 and message["role"] == "user"
                ]
                if result_positions:
                    compact = [dict(message) for message in round_messages]
                    overhead = _estimate([
                        {**message, "content": ""} if i in result_positions else message
                        for i, message in enumerate(compact)
                    ])
                    chars = (remaining - overhead) * 3 // len(result_positions) - 64
                    if chars > 0:
                        for i in result_positions:
                            compact[i]["content"] = excerpt(compact[i]["content"], chars)
                        if _estimate(compact) <= remaining:
                            recent_rounds.append(compact)
                            remaining -= _estimate(compact)
            break
        recent_rounds.append(round_messages)
        remaining -= cost

    recent_history = []
    for turn in reversed(_history_turns(history)):
        cost = _estimate(turn)
        if cost > remaining:
            break
        recent_history.append(turn)
        remaining -= cost

    return [
        *system,
        *(message for turn in reversed(recent_history) for message in turn),
        user_turn,
        *(message for round_messages in reversed(recent_rounds) for message in round_messages),
        *suffix,
    ]
