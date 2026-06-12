# ai/workspace

Lightweight local AI workspace. Chat + Agent + Memory. Works on Linux and Windows.

## Requirements
- Python 3.10+
- Ollama running on localhost:11434

## Setup

### Linux / macOS
```bash
git clone git@github.com:arnemar/Hangar.git
cd ai_workspace
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Windows (PowerShell)
```powershell
git clone git@github.com:arnemar/Hangar.git
cd ai_workspace
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:8080

## Config (environment variables)
- `OLLAMA_URL` — Ollama endpoint (default: http://localhost:11434)
- `DEFAULT_MODEL` — Default model (default: deepseek-coder-v2:16b)

## Auto-start on Linux (systemd)
```ini
[Unit]
Description=AI Workspace
After=network.target ollama.service

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/ai_workspace
ExecStart=/home/YOUR_USER/ai_workspace/venv/bin/python app.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

## Features
- **Chat** — streaming responses, session history, memory injection
- **Agent** — real tool execution (shell, file read/write, web search, memory)
- **Memory** — persistent facts injected into every session
- **Models** — switch models per session from sidebar

## Tools available to agent
- `shell` — run any shell command
- `read_file` — read file contents
- `write_file` — write/append to files
- `list_dir` — list directory contents
- `web_search` — DuckDuckGo search
- `manage_memory` — add/list/delete memories
