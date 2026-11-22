"""Read and atomically update local JSON state."""

import json
import os
import tempfile
from pathlib import Path

from hangar.config import DEFAULT_MODEL, MEMORY_PATH, OLLAMA_URL, PROJECTS_PATH, SETTINGS_PATH

_DEFAULTS = {
    "ollama_url": OLLAMA_URL,
    "default_model": DEFAULT_MODEL,
    "context_length": 8192,
    "system_prompt_chat": "",
    "use_saved_memory": False,
}


class StoreError(RuntimeError):
    pass


def _read(path: Path, default):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise StoreError(f"Hangar could not read {path.name}. Check the saved file.") from exc
    if not isinstance(data, type(default)):
        raise StoreError(f"{path.name} has an unexpected format.")
    return data


def _write(path: Path, value) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".hangar-", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise StoreError(f"Hangar could not save {path.name}. Check the data folder permissions.") from exc
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def load_memory() -> dict:
    return _read(MEMORY_PATH, {"facts": []})


def saved_facts() -> list[str]:
    facts = load_memory().get("facts")
    return [
        str(fact["content"]) for fact in facts
        if isinstance(fact, dict) and fact.get("content")
    ] if isinstance(facts, list) else []


def load_settings() -> dict:
    settings = {**_DEFAULTS, **_read(SETTINGS_PATH, {})}
    url = settings["ollama_url"]
    settings["ollama_url"] = (url.strip() if isinstance(url, str) else "") or OLLAMA_URL
    return settings


def save_settings(settings: dict) -> None:
    _write(SETTINGS_PATH, settings)


def load_projects() -> list[dict]:
    records = _read(PROJECTS_PATH, [])
    if any(not isinstance(record, dict) for record in records):
        raise StoreError("projects.json contains an invalid project record.")
    return records


def save_projects(projects: list[dict]) -> None:
    _write(PROJECTS_PATH, projects)


def get_project(name: str) -> dict | None:
    return next((record for record in load_projects() if record.get("name") == name), None)
