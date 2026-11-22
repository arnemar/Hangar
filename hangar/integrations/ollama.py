"""Ollama model discovery and validated JSON-line chat transport."""

import json
from typing import AsyncIterator

import httpx

from hangar.config import OLLAMA_URL


class OllamaError(RuntimeError):
    pass


def endpoint(address: str, path: str) -> str:
    return (address or OLLAMA_URL).rstrip("/") + "/api/" + path


def check_status(response: httpx.Response) -> None:
    if not response.is_error:
        return
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = body.get("error") if isinstance(body, dict) else None
    raise OllamaError(f"Ollama: {detail}" if isinstance(detail, str) and detail
                      else f"Ollama returned HTTP {response.status_code}.")


def object_payload(value) -> dict:
    if not isinstance(value, dict):
        raise OllamaError("Ollama returned an invalid response.")
    return value


async def request_object(address: str, path: str, payload: dict | None = None) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
            if payload is None:
                response = await client.get(endpoint(address, path))
            else:
                response = await client.post(endpoint(address, path), json=payload)
            check_status(response)
            return object_payload(response.json())
    except (httpx.HTTPError, httpx.InvalidURL) as error:
        raise OllamaError("Could not reach the configured Ollama server.") from error
    except ValueError as error:
        raise OllamaError("Ollama returned an invalid response.") from error


async def get_models(address: str) -> list[str]:
    data = await request_object(address, "tags")
    if not isinstance(data.get("models"), list):
        raise OllamaError("The configured server did not return an Ollama model list.")
    names = [
        item.get("name") or item.get("model") for item in data["models"] if isinstance(item, dict)
    ]
    return list(dict.fromkeys(name for name in names if isinstance(name, str) and name))


async def model_supports_tools(model: str, address: str) -> bool:
    data = await request_object(address, "show", {"model": model})
    capabilities = data.get("capabilities")
    return isinstance(capabilities, list) and "tools" in capabilities


def chat_frame(line: str) -> dict:
    frame = object_payload(json.loads(line))
    if frame.get("error"):
        raise OllamaError(f"Ollama: {frame['error']}")
    message = object_payload(frame.get("message", {}))
    for field in ("content", "thinking"):
        if field in message and not isinstance(message[field], str):
            raise OllamaError("Ollama returned an invalid chat message.")
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list) or any(
        not isinstance(call, dict) or not isinstance(call.get("function"), dict)
        or not isinstance(call["function"].get("name"), str)
        for call in calls
    ):
        raise OllamaError("Ollama returned an invalid tool call.")
    return frame


async def stream_ollama(
    messages: list[dict], model: str, tools: list[dict], num_ctx: int, address: str,
    generation: dict[str, int | float] | None = None,
) -> AsyncIterator[dict]:
    payload = {
        "model": model, "messages": messages, "stream": True,
        "options": {"num_ctx": num_ctx, **(generation or {})},
    }
    if tools:
        payload["tools"] = tools

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5)) as client:
            async with client.stream("POST", endpoint(address, "chat"), json=payload) as response:
                if response.is_error:
                    await response.aread()
                check_status(response)
                async for line in response.aiter_lines():
                    if line.strip():
                        frame = chat_frame(line)
                        yield frame
                        if frame.get("done") is True:
                            return
    except httpx.TimeoutException as error:
        raise OllamaError("Timed out waiting for Ollama.") from error
    except (httpx.HTTPError, httpx.InvalidURL) as error:
        raise OllamaError("The Ollama connection failed. Check the server and try again.") from error
    except (ValueError, RecursionError) as error:
        raise OllamaError("Ollama returned invalid streaming data.") from error
    raise OllamaError("Ollama ended the response before sending a completion message.")
