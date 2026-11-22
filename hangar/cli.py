"""Run the web workspace on the loopback interface."""

import argparse

import uvicorn

from hangar.config import BASE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hangar workspace")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true", help="Restart when Hangar's Python files change")
    args = parser.parse_args()
    uvicorn.run(
        "hangar.api.app:app", host=args.host, port=args.port, reload=args.reload,
        reload_dirs=[str(BASE_DIR / "hangar")] if args.reload else None,
        workers=1, proxy_headers=False,
    )
