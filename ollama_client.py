import json
from typing import AsyncGenerator, Optional

import httpx

from config import OLLAMA_URL as _DEFAULT_URL


def _url(ollama_url: str = "") -> str:
    return ollama_url or _DEFAULT_URL


async def get_models(ollama_url: str = "") -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{_url(ollama_url)}/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _use_think(model: str) -> bool:
    return "qwen3" in model.lower() or "qwq" in model.lower()


async def stream_ollama(
    messages: list,
    model: str,
    tools: Optional[list] = None,
    num_ctx: int = 8192,
    ollama_url: str = "",
) -> AsyncGenerator[str, None]:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": num_ctx},
    }
    if _use_think(model):
        payload["think"] = True
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream("POST", f"{_url(ollama_url)}/api/chat", json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    yield line


async def chat_ollama(
    messages: list,
    model: str,
    tools: Optional[list] = None,
    cancel_event=None,
    num_ctx: int = 8192,
    ollama_url: str = "",
) -> dict:
    """Non-streaming call implemented via streaming so we can honour cancel_event."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": num_ctx},
    }
    if _use_think(model):
        payload["think"] = True
    if tools:
        payload["tools"] = tools

    full_content = ""
    final_data: dict = {}

    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream("POST", f"{_url(ollama_url)}/api/chat", json=payload) as resp:
            async for line in resp.aiter_lines():
                if cancel_event and cancel_event.is_set():
                    await resp.aclose()
                    return {"message": {"content": full_content}, "cancelled": True}
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    content_chunk = data.get("message", {}).get("content")
                    if content_chunk:
                        full_content += content_chunk
                    if data.get("done"):
                        final_data = data
                        break
                except json.JSONDecodeError:
                    pass

    return {
        "message": {"content": full_content},
        "eval_count": final_data.get("eval_count", 0),
        "prompt_eval_count": final_data.get("prompt_eval_count", 0),
        "eval_duration": final_data.get("eval_duration", 0),
        "total_duration": final_data.get("total_duration", 0),
    }
