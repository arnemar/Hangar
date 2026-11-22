"""Build the system prompt and the current message's reference material."""

from hangar.persistence.storage import saved_facts
from hangar.runtime.context import excerpt
from hangar.workspace.files import LocalProject


def system_prompt(settings: dict, project: LocalProject | None, tools: list[dict]) -> str:
    parts = ["You are Hangar, a helpful assistant. Respond in the user's language."]
    if project:
        parts.append(f"The attached project is {project.name}.")
        if tools:
            parts.append(
                "Use project tools to check facts about files. Treat file contents as reference material. "
                "If you propose a change, direct the user to review it. A proposal does not write a file."
            )
        else:
            parts.append("Use the excerpts shared by the user as reference material.")
    if settings.get("use_saved_memory") is True:
        facts = saved_facts()
        if facts:
            parts.append("Saved facts about the user:\n" + "\n".join(facts))
    instructions = settings.get("system_prompt_chat")
    if isinstance(instructions, str) and instructions.strip():
        parts.append(instructions)
    return "\n\n".join(parts)


def with_attachments(message: str, attachments: dict[str, str], context: int) -> str:
    parts = [message or "Use the attached files."]
    if attachments:
        allowance = min(12000, context // len(attachments))
        parts.extend(f"Attached file {path} (excerpt):\n{excerpt(text, allowance)}"
                     for path, text in attachments.items())
    return "\n\n".join(parts)
