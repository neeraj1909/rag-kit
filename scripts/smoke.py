"""Run the public HTTP readiness, index, and cited-answer smoke journey."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit


class SmokeFailure(Exception):
    """A bounded public-contract smoke failure."""


def _require_http_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None:
        raise SmokeFailure("smoke URL must be an HTTP(S) address without user information")


def _request(
    url: str,
    *,
    request_id: str,
    body: dict[str, str] | None = None,
    timeout: float,
) -> dict[str, Any]:
    _require_http_url(url)
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json", "X-Request-ID": request_id}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET"
    )
    try:
        # _require_http_url rejects file/custom schemes and embedded credentials.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            payload = json.load(response)
            response_id = response.headers.get("X-Request-ID")
    except urllib.error.HTTPError as error:
        raise SmokeFailure(f"{url} returned HTTP {error.code}") from error
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise SmokeFailure(f"{url} did not return valid JSON") from error
    if not isinstance(payload, dict):
        raise SmokeFailure(f"{url} returned a non-object payload")
    if payload.get("schema_version") != "v1":
        raise SmokeFailure(f"{url} returned an unexpected schema version")
    if payload.get("request_id") != request_id or response_id != request_id:
        raise SmokeFailure(f"{url} did not preserve the request correlation ID")
    return payload


def _assert_cited_answer(payload: dict[str, Any]) -> str:
    answer = payload.get("answer")
    if not isinstance(answer, str) or "Cobalt Observatory" not in answer:
        raise SmokeFailure("ask response did not contain the fixture answer")
    citations = payload.get("citations")
    if not isinstance(citations, list) or not citations:
        raise SmokeFailure("ask response did not contain a citation")
    first = citations[0]
    if not isinstance(first, dict) or first.get("rank") != 1:
        raise SmokeFailure("ask response did not rank the first citation first")
    evidence = first.get("evidence")
    if not isinstance(evidence, list) or not evidence or not isinstance(evidence[0], dict):
        raise SmokeFailure("first citation did not contain provenance evidence")
    source_uri = evidence[0].get("source_uri")
    locator = evidence[0].get("locator")
    if not isinstance(source_uri, str) or not source_uri.endswith("/answer.txt"):
        raise SmokeFailure("first citation did not identify the fixture source")
    if not isinstance(locator, dict) or locator.get("kind") != "text_span":
        raise SmokeFailure("first citation did not preserve a text-span locator")
    return source_uri


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--source-uri", default="/data/corpus")
    parser.add_argument("--query", default="What is the fixture answer?")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--expect-manifest")
    args = parser.parse_args(argv)
    base_url = args.base_url.rstrip("/")
    try:
        ready = _request(f"{base_url}/readyz", request_id="smoke-ready", timeout=args.timeout)
        if ready.get("status") != "ready":
            raise SmokeFailure("readiness endpoint did not report ready")
        indexed = _request(
            f"{base_url}/v1/index",
            request_id="smoke-index",
            body={"source_uri": args.source_uri},
            timeout=args.timeout,
        )
        manifest = indexed.get("index_manifest_fingerprint")
        if (
            not isinstance(indexed.get("documents"), int)
            or indexed["documents"] <= 0
            or not isinstance(indexed.get("chunks"), int)
            or indexed["chunks"] <= 0
            or not isinstance(manifest, str)
            or not manifest
        ):
            raise SmokeFailure("index response did not report indexed documents and chunks")
        if args.expect_manifest is not None and manifest != args.expect_manifest:
            raise SmokeFailure("index manifest changed across the restart")
        answered = _request(
            f"{base_url}/v1/ask",
            request_id="smoke-ask",
            body={"query": args.query, "source_uri": args.source_uri},
            timeout=args.timeout,
        )
        citation_source = _assert_cited_answer(answered)
    except SmokeFailure as error:
        print(f"smoke failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "citation_source": citation_source,
                "documents": indexed["documents"],
                "index_manifest_fingerprint": manifest,
                "status": "ok",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
