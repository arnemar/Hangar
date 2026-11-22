"""Compatibility entry point for the Hangar web server."""

from hangar.api.app import app
from hangar.cli import main


if __name__ == "__main__":
    main()
