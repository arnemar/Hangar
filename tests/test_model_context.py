"""Regression coverage for prompt fitting."""

from hangar.runtime.context import fit_messages


def test_fit_keeps_current_request_and_most_recent_complete_turn():
    old_user = {"role": "user", "content": "old " * 1500}
    old_reply = {"role": "assistant", "content": "old reply"}
    recent_user = {"role": "user", "content": "last question"}
    recent_reply = {"role": "assistant", "content": "last reply"}
    current = {"role": "user", "content": "What about the second option?"}

    messages = fit_messages(
        "Instructions", [old_user, old_reply, recent_user, recent_reply], current, [], 512,
    )

    assert messages[1:] == [recent_user, recent_reply, current]


def test_fit_keeps_latest_tool_call_and_reply_together():
    call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file"}}]}
    output = {"role": "tool", "tool_name": "read_file", "content": "START " + "x" * 12000 + " END"}
    current = {"role": "user", "content": "Read the file"}

    messages = fit_messages("Instructions", [], current, [[call, output]], 1024)

    assert messages[1:3] == [current, call]
    assert messages[-1]["tool_name"] == "read_file"
    assert messages[-1]["content"].startswith("START ")
    assert messages[-1]["content"].endswith(" END")
    assert "characters omitted" in messages[-1]["content"]
    assert len(output["content"]) == 12010
