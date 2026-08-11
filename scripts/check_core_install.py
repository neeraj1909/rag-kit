#!/usr/bin/env python3
"""Prove the installed core is lightweight and reports optional installs."""

from __future__ import annotations

import contextlib
import io

import ragkit
from ragkit.delivery.server import main as server_main
from ragkit.infrastructure.optional import OptionalCapability, inspect_optional_capability

MISSING_BOUNDARIES = {
    "ocr": "pytesseract",
    "layout": "openpyxl",
    "vision": "transformers",
    "media": "faster_whisper",
    "persistent": "chromadb",
    "hosted": "openai",
    "http": "uvicorn",
    "reranking": "transformers",
}


def main() -> int:
    if not ragkit.__doc__:
        raise RuntimeError("ragkit package metadata is unavailable")
    for extra, module in MISSING_BOUNDARIES.items():
        result = inspect_optional_capability(OptionalCapability(extra, module))
        if result.installed:
            raise RuntimeError(f"core-only environment unexpectedly contains {module}")
        expected = f"install rag-kit[{extra}]"
        if result.action != expected:
            raise RuntimeError(
                f"{extra} diagnostic differs: {result.action!r}, expected {expected!r}"
            )
    error_output = io.StringIO()
    try:
        with contextlib.redirect_stderr(error_output):
            server_main(["--config", "/nonexistent/ragkit.toml"])
    except SystemExit as error:
        if error.code != 2 or "rag-kit[http]" not in error_output.getvalue():
            raise RuntimeError("HTTP server missing-extra diagnostic is not actionable") from error
    else:
        raise RuntimeError("core-only HTTP server unexpectedly started without the http extra")
    print("core import is lightweight; missing-extra diagnostics are actionable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
