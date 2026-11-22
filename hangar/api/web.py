"""Serve the browser files that were present when the API started."""

import hashlib
from pathlib import Path

from starlette.responses import HTMLResponse, Response

MEDIA_TYPES = {".css": "text/css", ".js": "text/javascript"}


class WebBundle:
    def __init__(self, web: Path):
        root = web / "assets"
        self.files = {}
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.suffix not in MEDIA_TYPES or path.is_symlink() or not path.is_file():
                continue
            name = path.relative_to(root).as_posix()
            content = path.read_bytes()
            self.files[name] = (content, MEDIA_TYPES[path.suffix])
            digest.update(name.encode("utf-8") + b"\0" + hashlib.sha256(content).digest())
        self.version = digest.hexdigest()[:16]
        self.page = (web / "index.html").read_text(encoding="utf-8").replace(
            '"/static/', f'"/assets/{self.version}/',
        )

    async def index(self, request):
        return HTMLResponse(self.page, headers={"Cache-Control": "no-store"})

    async def asset(self, request):
        version = request.path_params.get("version")
        entry = self.files.get(request.path_params["asset"])
        if entry is None or version not in (None, self.version):
            return Response(status_code=404, headers={"Cache-Control": "no-store"})
        content, media_type = entry
        cache = "public, max-age=31536000, immutable" if version else "no-cache"
        return Response(content, media_type=media_type, headers={"Cache-Control": cache})
