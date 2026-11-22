# Hangar

Hangar is a local Ollama chat workspace. You can attach a project folder, share file excerpts, and review proposed edits as diffs.

## Run Hangar

Install Python 3.10 or later. Start Ollama and make sure it has at least one chat model installed. Ollama can run locally or on a machine Hangar can reach.

```sh
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux or macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). Use **Settings** to enter a different Ollama address and **Refresh models** after installing a model. Use `python app.py --port 8081` for a different port, or add `--reload` during development. The browser UI requires no build step.

## Work with a project

- **Attach project** registers an absolute folder path on the machine running Hangar. Browse UTF-8 files and attach up to three excerpts per message. Models that support Ollama tool calling can also browse or read the attached project on request.
- **Suggest edits** lets a tool-capable model propose a text replacement or new file. Open **Changes** to review the diff and apply or dismiss it. If the source file changes before approval, request a fresh proposal.
- **Search web** sends the typed question to DuckDuckGo. Hangar adds any file excerpts separately when it sends the prompt to Ollama.
- **Settings** controls the Ollama URL, requested context window, custom instructions, and whether to include saved facts.

Hangar binds to loopback. Its project browser skips links, common credential files, and Hangar's data directory. File previews and proposals accept UTF-8 text files up to 256 KiB. New files need an existing parent folder.

Chat history, settings, and project registrations continue to use `data/`. To run a separate instance with its own data, set `HANGAR_DATA_DIR` before starting Hangar. Older chats open in the same history. SSH project registrations are shown as unavailable in the local picker.

## Layout

| Path | Responsibility |
| --- | --- |
| `hangar/api/` | HTTP routes, stream lifecycle, and browser access |
| `hangar/integrations/` | Ollama transport and response validation |
| `hangar/runtime/` | Chat turns, prompt context, and web search |
| `hangar/workspace/` | Scoped file access, model tools, and edit review |
| `hangar/persistence/` | SQLite history and JSON settings |
| `hangar/web/` | Browser modules and local styles |
| `hangar/compiler/` | Standalone code graph and retrieval |

## Verify

```sh
python -m pip install -r requirements-dev.txt
python -m pytest
```

The API tests simulate Ollama and cover saved chat, disconnects, and reviewed writes. To include the retrieval benchmark, generate its fixture first with `python tools/build_graph_fixture.py`.
