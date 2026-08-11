# HTTP delivery and observability

The HTTP surface is an optional delivery adapter. It maps exact JSON requests
to `IndexingRequest` and `AnsweringRequest`, calls the existing application
pipeline once, and serializes the typed result. Retrieval, ranking, prompting,
generation, provenance, and index compatibility remain application and domain
responsibilities.

## Run the server

Install the `http` extra and pass one explicit profile:

```bash
uv sync --extra http
uv run ragkit-http --config examples/assignment_profiles/local-offline/ragkit.toml \
  --host 127.0.0.1 --port 8000
```

The launcher loads and composes the profile before it starts listening. A
missing server dependency fails with the exact `rag-kit[http]` installation
instruction. Uvicorn access logging is disabled; ragkit emits the bounded JSON
events described below.

## Routes and schemas

Every response is JSON and carries `schema_version: "v1"`, `request_id`, and an
`X-Request-ID` header. A caller-supplied ID is retained only when it matches
`[A-Za-z0-9._-]{1,64}`; otherwise the adapter generates one. Request bodies are
limited to 16 KiB and unknown fields are rejected.

| Route | Exact request | Successful response fields |
|---|---|---|
| `GET /healthz` | no body | `status: "ok"` |
| `GET /readyz` | no body | `status: "ready"` |
| `POST /v1/index` | `{"source_uri":"..."}` | `documents`, `chunks`, `index_manifest_fingerprint`, `diagnostics` |
| `POST /v1/ask` | `{"query":"...","source_uri":"..."}` | `answer`, `citations`, `model_fingerprint`, `diagnostics` |

Index a source before asking against it. This is observable for process-local
profiles and required for a new persistent collection. The ask route does not
silently index, rebuild, or migrate data.

`source_uri` must resolve to the source configured in the composed profile.
Equivalent path spellings are canonicalized to that configured value; another
host path fails with `source_not_allowed` before manifest construction or
connector access.

Errors have one stable shape:

```json
{"error":{"code":"invalid_request","message":"request is invalid"},"request_id":"...","schema_version":"v1"}
```

Error messages are deliberately generic. Exception messages can contain paths,
provider payloads, or credentials and are neither returned nor logged.

## Structured telemetry and redaction

`JsonLinesTelemetry` writes one object per event with only:

- `operation`, `outcome`, and `duration_ns`;
- bounded scalar `attributes` approved by the caller.

Application-stage events carry a component fingerprint, result count, and a
stable error category (`none` on success). Components with a public behavioral
fingerprint use it directly; otherwise the application records a deterministic
implementation-identity fingerprint. Unexpected exception classes collapse to
`unexpected_error`, so provider exception text and class details do not cross
the telemetry boundary.

The HTTP launcher injects one request-correlated sink into both the application
pipeline and the delivery adapter. Every nested application-stage event carries
the same bounded `request_id` as its terminal `http.request` event. Correlation
uses request-local context, so concurrent requests do not share identifiers.

HTTP request events contain `method`, normalized route, `status_code`, and
`request_id`. They never contain a request body, source URI, query, answer,
prompt, document text, exception message, credential, or token. Attribute count
and string lengths are bounded; duplicate names and sensitive-name fragments
are rejected before a line is written. `InMemoryTelemetry` applies the same
redaction contract.

The JSON-lines sink is serialized with a lock and flushes each complete line.
It does not promise durable storage after the process or underlying stream
fails. Replace the `Telemetry` port at composition when durable export is
required; keep the same bounded, content-free event contract.

## Operational meaning

- Liveness means the ASGI process can serve a request.
- Readiness means profile loading and runtime composition succeeded and the
  injected readiness predicate currently passes.
- Readiness does not claim that a hosted provider, optional model, or external
  database is reachable. Those capabilities still fail explicitly when used.
- Telemetry durations are monotonic process timings, not quality, confidence,
  or service-level-objective claims.
