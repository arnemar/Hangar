"""The local Hangar web application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from hangar.api.routes import changes, chat, models, projects, sessions, settings
from hangar.persistence.conversations import migrate_workspace
from hangar.persistence.db import init_chat_db
from hangar.persistence.storage import StoreError
from hangar.workspace.files import WorkspaceError

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_chat_db()
    migrate_workspace()
    yield


app = FastAPI(title="Hangar", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def same_origin(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if (origin is not None and origin != expected) or request.headers.get("sec-fetch-site") in {
            "cross-site", "same-site",
        }:
            return JSONResponse({"detail": "Open Hangar directly to use its local API."}, status_code=403)
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(WorkspaceError)
async def workspace_error(request: Request, error: WorkspaceError):
    return JSONResponse({"detail": str(error)}, status_code=error.status)


@app.exception_handler(StoreError)
async def stored_data_error(request: Request, error: StoreError):
    return JSONResponse({"detail": str(error)}, status_code=409)


for feature in (models, sessions, chat, settings, projects, changes):
    app.include_router(feature.router)


@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static/css", StaticFiles(directory=WEB_DIR / "assets" / "css"), name="css")
app.mount("/static/js", StaticFiles(directory=WEB_DIR / "assets" / "js"), name="js")
