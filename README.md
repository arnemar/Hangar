# Hangar

Hangar is a local chat app for Ollama. Start a conversation, attach a project for file context, and review proposed changes as diffs.

## Run Hangar

Install Python 3.10+ and start Ollama. Ollama can run on this computer or on a separate machine reachable from it.

1. Create a virtual environment with `python -m venv .venv` and activate it.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Run `python app.py` and open [127.0.0.1:8080](http://127.0.0.1:8080).

The frontend uses browser modules and needs no build or CDN. Use `python app.py --reload` for development.

## Chat, files, and changes

- **Chat.** Choose an installed model per conversation. Set your Ollama URL, context window, and custom instructions in Settings. Saved facts from older versions can be reviewed and enabled there.
- **Projects.** Attach a registered local folder by its absolute path on the computer running Hangar. Browse and preview UTF-8 files, or attach up to three excerpts to a message. Models with native Ollama tool calling can also list and read project files. For other models, use the attachment button.
- **Edits.** Enable **Suggest edits** for the message. Review each proposed replacement or new file under **Changes**, and click **Apply change** to write it. A change applies only to its original project and file contents. Ask for a new proposal if the file has changed.
- **Web search.** Enable **Search web** for the message to send the typed question to DuckDuckGo. Attached file contents stay out of that lookup.

Hangar serves loopback only. The project browser rejects paths outside the attached folder and skips linked files, common credential files, and Hangar's data directory. Text file previews and edits support UTF-8 files up to 256 KiB. New files need an existing parent folder. Shell, SSH, MCP, and command execution are outside the chat tool set.

Conversation history, project registrations, and settings use the existing `data/` directory. `HANGAR_DATA_DIR` can point to a separate directory for a test instance. Older chats open in the unified chat view. Legacy SSH project records remain stored and are unavailable in the local project picker.

## Project layout

| Path | Responsibility |
| --- | --- |
| `hangar/api/` | Local web server and API |
| `hangar/runtime/` | Chat orchestration and context |
| `hangar/workspace/` | Project file access and approved edits |
| `hangar/persistence/` | SQLite, settings, and project registrations |
| `hangar/integrations/` | Ollama transport |
| `hangar/web/` | Browser UI and local assets |
| `hangar/compiler/` | Standalone code graph and retrieval engine |
| `tests/`, `tools/` | Regression tests and graph fixture builder |

## Tests

Install test dependencies with `python -m pip install -r requirements-dev.txt`. Run `python -m pytest` for the workspace tests. To include the compiler retrieval benchmark, run `python tools/build_graph_fixture.py` before pytest.
