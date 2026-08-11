#!/usr/bin/env python3
"""Dependency-free readiness probe used inside the runtime image."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from urllib.parse import urlsplit


def _valid_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and parsed.username is None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args(argv)
    if not _valid_http_url(args.url):
        return 1
    try:
        # The scheme and authority are validated immediately above.
        with urllib.request.urlopen(args.url, timeout=args.timeout) as response:  # nosec B310
            if response.status != 200:
                return 1
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return 1
    valid = (
        isinstance(payload, dict)
        and payload.get("schema_version") == "v1"
        and payload.get("status") == "ready"
        and isinstance(payload.get("request_id"), str)
        and bool(payload["request_id"])
    )
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
