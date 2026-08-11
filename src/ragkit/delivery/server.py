"""Optional Uvicorn launcher for the ragkit ASGI delivery adapter."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from ragkit.adapters import JsonLinesTelemetry, RequestCorrelatedTelemetry
from ragkit.infrastructure import bootstrap, load_config

from .http import create_app


class _Uvicorn(Protocol):
    def run(
        self, app: object, *, host: str, port: int, access_log: bool, lifespan: str
    ) -> None: ...


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragkit-http")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load one profile and run the optional HTTP server."""

    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        _parser().error("--port must be between 1 and 65535")
    try:
        uvicorn = cast(_Uvicorn, import_module("uvicorn"))
    except ModuleNotFoundError:
        _parser().error("HTTP serving requires: pip install 'rag-kit[http]'")
    profile = load_config(args.config)
    telemetry = RequestCorrelatedTelemetry(JsonLinesTelemetry())
    runtime = bootstrap(profile, telemetry=telemetry)
    app = create_app(runtime, profile, telemetry)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, lifespan="off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
