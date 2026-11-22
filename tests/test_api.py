"""Exercise chat, tool calls, reviewed writes, and failures through the HTTP API."""

import asyncio
import json
import re
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import httpx

from hangar.api.app import create_app
from hangar.integrations import ollama
from hangar.persistence import db, storage
from hangar.workspace import files


class OllamaFixture:
    def __init__(self):
        self.requests = []
        self.response = [self.frame("Ready."), {"done": True, "done_reason": "stop"}]

    @staticmethod
    def frame(content="", **fields):
        return {"message": {"role": "assistant", "content": content, **fields}, "done": False}

    def __call__(self, request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        if request.url.path == "/api/chat":
            payload = json.loads(request.content)
            self.requests.append(payload)
            result = self.response(payload) if callable(self.response) else self.response
            if isinstance(result, httpx.Response):
                return result
            return httpx.Response(200, text="\n".join(json.dumps(frame) for frame in result))
        return httpx.Response(404)


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


class TestWorkspaceAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.project = root / "workspace"
        self.project.mkdir()
        self.note = self.project / "notes.txt"
        self.note.write_text("original\n", encoding="utf-8")
        stack.enter_context(patch.object(files, "DATA_DIR", root / "data"))
        stack.enter_context(patch.object(db, "DB_PATH", root / "data" / "workspace.db"))
        for name in ("PROJECTS", "SETTINGS", "MEMORY"):
            stack.enter_context(patch.object(storage, name + "_PATH", root / "data" / (name.lower() + ".json")))

        self.app = create_app()
        lifespan = self.app.router.lifespan_context(self.app)
        await lifespan.__aenter__()
        self.addAsyncCleanup(lifespan.__aexit__, None, None, None)
        client_cls = httpx.AsyncClient
        self.client = client_cls(transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1")
        self.addAsyncCleanup(self.client.aclose)
        self.ollama = OllamaFixture()
        stack.enter_context(patch.object(
            ollama.httpx, "AsyncClient",
            side_effect=lambda **kwargs: client_cls(transport=httpx.MockTransport(self.ollama), **kwargs),
        ))

    async def create_chat(self, project=None):
        response = await self.client.post("/api/sessions", json={"model": "test-model", "project_name": project})
        self.assertEqual(response.status_code, 201, response.text)
        return "/api/sessions/" + response.json()["id"]

    async def test_stream_and_saved_history(self):
        page = await self.client.get("/")
        self.assertEqual(page.status_code, 200)
        asset_links = re.findall(r'"(/assets/[^"]+)"', page.text)
        self.assertGreater(len(asset_links), 1)
        for url in asset_links:
            asset = await self.client.get(url)
            self.assertEqual(asset.status_code, 200, url)
            self.assertIn("immutable", asset.headers["cache-control"])
        models = (await self.client.get("/api/models")).json()
        self.assertEqual(models["default"], "test-model")

        chat = await self.create_chat()
        response = await self.client.post(chat + "/chat", json={"message": "Hello"})
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        self.assertEqual([event["type"] for event in events(response)], ["started", "activity", "chunk", "done"])
        saved = (await self.client.get(chat + "/messages")).json()["messages"]
        self.assertEqual([message["content"] for message in saved], ["Hello", "Ready."])
        self.assertEqual((await self.client.get("/api/sessions")).json()["sessions"][0]["title"], "Hello")

        await self.client.post(chat + "/chat", json={"message": "Continue"})
        history = self.ollama.requests[-1]["messages"]
        self.assertEqual([message["content"] for message in history[1:]], ["Hello", "Ready.", "Continue"])

    async def test_running_app_serves_one_version_of_the_web_files(self):
        web = self.project / "web"
        assets = web / "assets" / "js"
        assets.mkdir(parents=True)
        index = web / "index.html"
        index.write_text('<script src="/static/js/main.js"></script>', encoding="utf-8")
        script = assets / "main.js"
        script.write_text("console.log('before')", encoding="utf-8")
        with patch("hangar.api.app.WEB_DIR", web):
            running = create_app()
            script.write_text("console.log('after')", encoding="utf-8")
            index.write_text('<title>New</title><script src="/static/js/main.js"></script>', encoding="utf-8")
            replacement = create_app()

        # Reuse the real client type because the fixture patches Ollama's HTTP client.
        client_type = type(self.client)
        async with (
            client_type(transport=httpx.ASGITransport(app=running), base_url="http://127.0.0.1") as old,
            client_type(transport=httpx.ASGITransport(app=replacement), base_url="http://127.0.0.1") as new,
        ):
            before = await old.get("/")
            after = await new.get("/")
            before_url = re.search(r'src="([^"]+)"', before.text)[1]
            after_url = re.search(r'src="([^"]+)"', after.text)[1]
            self.assertEqual(before.headers["cache-control"], "no-store")
            self.assertNotIn("New", before.text)
            self.assertIn("New", after.text)
            self.assertNotEqual(before_url, after_url)
            self.assertIn("before", (await old.get(before_url)).text)
            self.assertIn("after", (await new.get(after_url)).text)
            self.assertEqual((await new.get(before_url)).status_code, 404)

    async def test_attached_file_proposal_requires_review(self):
        registered = await self.client.post("/api/projects", json={"name": "Notes", "root": str(self.project)})
        self.assertEqual(registered.status_code, 201, registered.text)
        chat = await self.create_chat("Notes")
        listed = (await self.client.get("/api/projects/Notes/files")).json()["entries"]
        self.assertEqual([item["path"] for item in listed], ["notes.txt"])
        outside = await self.client.get("/api/projects/Notes/file", params={"path": "../settings.json"})
        self.assertEqual(outside.status_code, 403)

        def answer(payload):
            if any(message["role"] == "tool" for message in payload["messages"]):
                return [self.ollama.frame("Review the proposed change."), {"done": True}]
            self.assertIn("original", payload["messages"][-1]["content"])
            call = {"function": {"name": "propose_file_change", "arguments": {
                "path": "notes.txt", "operation": "replace", "old_text": "original", "new_text": "updated",
            }}}
            return [self.ollama.frame("Drafting", tool_calls=[call]), {"done": True}]

        self.ollama.response = answer
        response = await self.client.post(chat + "/chat", json={
            "message": "Update the note", "attachments": ["notes.txt", "notes.txt"], "propose_edits": True,
        })
        timeline = events(response)
        self.assertIn("reset", [event["type"] for event in timeline])
        changes = (await self.client.get(chat + "/changes")).json()["changes"]
        self.assertEqual(len(changes), 1)
        self.assertIn("+updated", changes[0]["diff"])
        self.assertEqual(self.note.read_text(encoding="utf-8"), "original\n")
        reviewed = await self.client.post(chat + "/changes/" + changes[0]["id"], json={"decision": "apply"})
        self.assertEqual(reviewed.json()["status"], "applied")
        self.assertEqual(self.note.read_text(encoding="utf-8"), "updated\n")

    async def test_upstream_error_and_retry(self):
        chat = await self.create_chat()
        self.ollama.response = httpx.Response(500, json=["malformed error"])
        failed = await self.client.post(chat + "/chat", json={"message": "First attempt"})
        self.assertEqual(events(failed)[-1], {"type": "error", "content": "Ollama returned HTTP 500."})
        self.ollama.response = [self.ollama.frame("Partial")]
        truncated = await self.client.post(chat + "/chat", json={"message": "Again"})
        self.assertEqual(events(truncated)[-1]["type"], "error")
        self.ollama.response = [self.ollama.frame("Done"), {"done": True}]
        retried = await self.client.post(chat + "/chat", json={"message": "Retry"})
        self.assertEqual(events(retried)[-1]["type"], "done")
        saved = (await self.client.get(chat + "/messages")).json()["messages"]
        self.assertEqual([message["content"] for message in saved], ["First attempt", "Again", "Retry", "Done"])

    async def test_project_picker_browses_directories_and_searches_names(self):
        working = self.project.resolve() / "Working"
        child = working / "Example App"
        child.mkdir(parents=True)
        (self.project / ".ssh" / "Example private").mkdir(parents=True)

        picker = "/api/projects/folders"
        with patch("hangar.workspace.folders.Path.home", return_value=self.project):
            home = await self.client.get(picker)
        self.assertEqual(home.status_code, 200, home.text)
        self.assertEqual(home.json()["folders"], [{"name": "Working", "path": str(working)}])

        found = await self.client.get(picker, params={"path": str(self.project), "q": "example"})
        self.assertEqual(found.json()["folders"], [{"name": "Example App", "path": str(child)}])
        direct = await self.client.get(picker, params={"path": str(child)})
        self.assertEqual(direct.json()["folder"]["path"], str(child))
        self.assertEqual(direct.json()["parent"], str(working))
        blocked = await self.client.get(picker, params={"path": str(self.project / ".ssh")})
        self.assertEqual(blocked.status_code, 403)
        protected = await self.client.get(picker, params={"path": str(files.DATA_DIR)})
        self.assertEqual(protected.status_code, 403)
        self.assertEqual((await self.client.get("/api/projects")).json()["projects"], [])

    async def test_generation_settings_round_trip_to_ollama(self):
        endpoint = "/api/settings"
        previous = (await self.client.get(endpoint)).json()
        self.assertIsNone(previous["temperature"])
        settings = {
            "ollama_url": previous["ollama_url"],
            "default_model": "test-model",
            "context_length": 16384,
            "temperature": 0,
            "top_p": 0.8,
            "max_output_tokens": 2048,
        }
        saved = await self.client.put(endpoint, json=settings)
        self.assertEqual(saved.status_code, 200, saved.text)
        invalid = await self.client.put(endpoint, json={**settings, "top_p": True})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual((await self.client.get(endpoint)).json()["top_p"], 0.8)

        chat = await self.create_chat()
        await self.client.post(chat + "/chat", json={"message": "Hello"})
        self.assertEqual(self.ollama.requests[-1]["options"], {
            "num_ctx": 16384, "temperature": 0, "top_p": 0.8, "num_predict": 2048,
        })
        await self.client.put(endpoint, json={**settings, "temperature": None, "max_output_tokens": None})
        await self.client.post(chat + "/chat", json={"message": "Again"})
        self.assertEqual(self.ollama.requests[-1]["options"], {"num_ctx": 16384, "top_p": 0.8})

    async def test_rejects_cross_origin_and_invalid_json(self):
        chat = await self.create_chat()
        foreign = await self.client.delete(chat, headers={"origin": "http://example.com"})
        self.assertEqual(foreign.status_code, 403)
        wrong_type = await self.client.post(chat + "/chat", content="message")
        self.assertEqual(wrong_type.status_code, 415)
        malformed = await self.client.post(chat + "/chat", json={"attachments": [False]})
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(self.ollama.requests, [])

    async def test_disconnect_releases_chat_and_upstream_stream(self):
        chat = await self.create_chat()
        waiting = asyncio.Event()
        closed = asyncio.Event()

        class HangingResponse(httpx.AsyncByteStream):
            async def __aiter__(self):
                waiting.set()
                await asyncio.Future()
                yield b""

            async def aclose(self):
                closed.set()

        self.ollama.response = lambda _: httpx.Response(200, stream=HangingResponse())
        queued = asyncio.Queue()
        queued.put_nowait({
            "type": "http.request", "body": b'{"message":"Stop this"}', "more_body": False,
        })

        async def send(_):
            pass

        scope = {
            "type": "http", "asgi": {"spec_version": "2.3"}, "http_version": "1.1",
            "method": "POST", "path": chat + "/chat", "root_path": "", "scheme": "http",
            "query_string": b"", "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"application/json")],
        }
        stream = asyncio.create_task(self.app(scope, queued.get, send))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=2)
            busy = await self.client.patch(chat, json={"model": "other-model"})
            self.assertEqual(busy.status_code, 409)
            queued.put_nowait({"type": "http.disconnect"})
            await asyncio.wait_for(stream, timeout=2)
            self.assertTrue(closed.is_set())
            patch_session = await self.client.patch(chat, json={"model": "other-model"})
            self.assertEqual(patch_session.status_code, 200)
        finally:
            if not stream.done():
                stream.cancel()
                await asyncio.gather(stream, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
