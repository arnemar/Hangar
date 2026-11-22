"""Same-origin access and browser response headers for the loopback app."""

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse


class BrowserBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        is_api = scope["path"].startswith("/api/")
        headers = Headers(scope=scope)

        async def send_headers(message):
            if message["type"] == "http.response.start":
                response = MutableHeaders(scope=message)
                response["Content-Security-Policy"] = (
                    "default-src 'self'; object-src 'none'; base-uri 'none'; "
                    "frame-ancestors 'none'; form-action 'self'"
                )
                response["Referrer-Policy"] = "no-referrer"
                response["X-Content-Type-Options"] = "nosniff"
                if is_api:
                    response["Cache-Control"] = "no-store"
            await send(message)

        expected = f"{scope['scheme']}://{headers.get('host', '')}"
        origin = headers.get("origin")
        if is_api and (
            (origin is not None and origin != expected)
            or headers.get("sec-fetch-site") in {"cross-site", "same-site"}
        ):
            response = JSONResponse(
                {"detail": "Open Hangar directly to use its local API."}, status_code=403,
            )
            await response(scope, receive, send_headers)
            return
        await self.app(scope, receive, send_headers)
