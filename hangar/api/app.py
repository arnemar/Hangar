"""Assemble the local HTTP API and static browser application."""

from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from hangar.api.boundary import BrowserBoundary
from hangar.api.routes import changes, chat, models, projects, sessions, settings
from hangar.api.web import WebBundle
from hangar.persistence.conversations import migrate_workspace
from hangar.persistence.db import init_chat_db
from hangar.persistence.storage import StoreError
from hangar.workspace.files import WorkspaceError

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def lifespan(app):
    init_chat_db()
    migrate_workspace()
    yield


async def workspace_error(request, error):
    return JSONResponse({"detail": str(error)}, status_code=error.status)


async def storage_error(request, error):
    return JSONResponse({"detail": str(error)}, status_code=409)


def create_app() -> Starlette:
    web = WebBundle(WEB_DIR)
    app = Starlette(
        lifespan=lifespan,
        routes=[
            *(route for feature in (models, sessions, chat, settings, projects, changes)
              for route in feature.routes),
            Route("/", web.index),
            Route("/assets/{version}/{asset:path}", web.asset),
            Route("/static/{asset:path}", web.asset),
        ],
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"]),
            Middleware(BrowserBoundary),
        ],
        exception_handlers={WorkspaceError: workspace_error, StoreError: storage_error},
    )
    app.state.active_chats = set()
    return app


app = create_app()
