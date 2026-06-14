"""MCP (Model Context Protocol) stdio client.

Connects to external MCP servers, discovers their tools, and proxies
tool calls. Each server runs as a subprocess communicating via JSON-RPC
over stdin/stdout (MCP 2024-11-05 stdio transport).

Tool names are namespaced as  mcp__{server}__{tool}  to avoid collisions
with built-in tools.
"""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)


class MCPClient:
    """Connection to a single MCP server process."""

    def __init__(self, name: str, command: list[str], env: dict | None = None):
        self.name = name
        self.command = command
        self.env = env or {}
        self.process: asyncio.subprocess.Process | None = None
        self.tools: list[dict] = []
        self._id = 0
        self._lock = asyncio.Lock()

    async def start(self) -> tuple[bool, str]:
        try:
            merged_env = {**os.environ, **self.env}
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=merged_env,
            )

            # Handshake: initialize
            init = await self._rpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hangar", "version": "1.0"},
            })
            if "error" in init:
                return False, f"Init error: {init['error']}"

            # Notify server we're ready
            notif = json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }) + "\n"
            self.process.stdin.write(notif.encode())
            await self.process.stdin.drain()

            # Discover tools
            tools_resp = await self._rpc("tools/list", {})
            self.tools = tools_resp.get("result", {}).get("tools", [])
            logger.info("MCP '%s': connected, %d tools", self.name, len(self.tools))
            return True, f"{len(self.tools)} tools available"

        except FileNotFoundError:
            return False, f"Command not found: {self.command[0]}"
        except Exception as exc:
            return False, str(exc)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        try:
            resp = await self._rpc("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })
            r = resp.get("result", {})
            content = r.get("content", [])
            texts = [c["text"] for c in content if c.get("type") == "text"]
            if r.get("isError"):
                return f"[error]: {' '.join(texts) or 'tool error'}"
            return "\n".join(texts) if texts else "[no output]"
        except Exception as exc:
            return f"[error]: MCP call failed: {exc}"

    async def _rpc(self, method: str, params: dict) -> dict:
        async with self._lock:
            self._id += 1
            msg = json.dumps({
                "jsonrpc": "2.0",
                "id": self._id,
                "method": method,
                "params": params,
            }) + "\n"
            self.process.stdin.write(msg.encode())
            await self.process.stdin.drain()
            raw = await asyncio.wait_for(
                self.process.stdout.readline(), timeout=15.0
            )
            if not raw:
                raise RuntimeError("MCP server closed connection")
            return json.loads(raw)

    def is_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def stop(self):
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass


class MCPManager:
    """Registry of all active MCP server connections."""

    def __init__(self):
        self.clients: dict[str, MCPClient] = {}

    async def connect(
        self, name: str, command: list[str], env: dict | None = None
    ) -> tuple[bool, str]:
        """Start (or restart) a named MCP server."""
        if name in self.clients:
            await self.clients[name].stop()
            del self.clients[name]

        client = MCPClient(name, command, env)
        ok, msg = await client.start()
        if ok:
            self.clients[name] = client
        return ok, msg

    async def disconnect(self, name: str):
        if name in self.clients:
            await self.clients[name].stop()
            del self.clients[name]

    def get_tool_descriptions(self) -> list[str]:
        """One-liner descriptions for the agent system prompt."""
        lines = []
        for server_name, client in self.clients.items():
            if not client.is_alive():
                continue
            for t in client.tools:
                fn_name = f"mcp__{server_name}__{t['name']}"
                props = t.get("inputSchema", {}).get("properties", {})
                params = ", ".join(
                    f'{k}: {v.get("type", "any")}' for k, v in props.items()
                )
                lines.append(f"- {fn_name}({params}): [MCP:{server_name}] {t.get('description', '')}")
        return lines

    async def call_tool(self, prefixed_name: str, arguments: dict) -> str:
        """Dispatch a prefixed tool call to the correct MCP server."""
        if not prefixed_name.startswith("mcp__"):
            return f"[error]: Not an MCP tool: {prefixed_name}"
        rest = prefixed_name[5:]  # strip "mcp__"
        parts = rest.split("__", 1)
        if len(parts) < 2:
            return f"[error]: Invalid MCP tool name: {prefixed_name}"
        server_name, tool_name = parts
        client = self.clients.get(server_name)
        if not client or not client.is_alive():
            return f"[error]: MCP server '{server_name}' not connected"
        return await client.call_tool(tool_name, arguments)

    def get_status(self) -> list[dict]:
        return [
            {
                "name": name,
                "command": client.command,
                "connected": client.is_alive(),
                "tool_count": len(client.tools),
                "tools": [
                    {"name": t["name"], "description": t.get("description", "")}
                    for t in client.tools
                ],
            }
            for name, client in self.clients.items()
        ]

    async def disconnect_all(self):
        for client in self.clients.values():
            await client.stop()
        self.clients.clear()


# Singleton used by app.py and agent.py
mcp_manager = MCPManager()
