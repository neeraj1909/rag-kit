# Container deployment

The supported container is a CPU-only, offline-first HTTP deployment. It installs
only the `http` and `persistent` extras, runs as UID/GID `65532`, drops Linux
capabilities, and uses a read-only root filesystem. The published port binds to
loopback by default; put an authenticated reverse proxy in front of it before
exposing it outside one machine.

## Filesystem contract

The Compose service deliberately separates operator input from runtime state:

| Container path | Access | Purpose |
|---|---|---|
| `/app/config/ragkit.toml` | read-only bind mount | exact, secret-free runtime profile |
| `/data/corpus` | read-only bind mount | source documents visible to the connector |
| `/var/lib/ragkit` | writable named volume | Chroma index and manifest state |
| `/tmp` | bounded tmpfs | transient framework files |

`deployment/ragkit.toml` uses absolute container paths. Changing the corpus,
chunker, embedder, dimension, or schema can make an existing index incompatible;
ragkit rejects that mismatch rather than silently rebuilding the volume. Use a
new explicitly named Compose volume for a deliberately new index. Do not edit
files inside a managed volume by hand.

The image contains no corpus, model cache, credential, `.env` file, or index.
Hosted credentials are not part of this baseline. If a later deployment enables
a hosted adapter, inject its credential at runtime through the platform's secret
facility; never add it to the image, Compose file, or TOML profile.

## Chroma advisory boundary

The locked `chromadb` package is covered by
[GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c),
which has no patched Python release as of 2026-08-11. The vulnerable surface is
Chroma's unauthenticated FastAPI collection endpoint accepting a model repository
and `trust_remote_code`. This deployment does not start or publish that server:
rag-kit uses an in-process `PersistentClient`, creates a fixed cosine collection,
and exposes only its own exact-schema API. No request field can select a Chroma
model repository or `trust_remote_code` value. Do not add a Chroma server port or
run its Python FastAPI command. Re-run the dependency audit and upgrade as soon as
a reviewed patched release exists; this boundary is a mitigation, not a claim that
the installed distribution is vulnerability-free.

## Clean build and readiness

From the repository root:

```bash
docker compose build --no-cache
docker compose up -d --wait
docker compose ps
python scripts/container_health.py --url http://127.0.0.1:8000/readyz
python scripts/smoke.py --base-url http://127.0.0.1:8000
```

`docker compose up --wait` succeeds only after the application readiness route
reports that the configured profile can be composed. A running process is not
treated as readiness.

To prove the named index survives a process replacement, run the smoke first,
restart only the service, wait, and run it again:

```bash
first_smoke=$(python scripts/smoke.py --base-url http://127.0.0.1:8000)
manifest=$(printf '%s' "$first_smoke" | python -c \
  'import json,sys; print(json.load(sys.stdin)["index_manifest_fingerprint"])')
docker compose restart ragkit
docker compose up -d --wait
python scripts/smoke.py \
  --base-url http://127.0.0.1:8000 \
  --expect-manifest "$manifest"
```

The second response must remain cited and the service must retain persistent
index semantics. This is a restart/reopen proof, not a claim that a named volume
survives `docker compose down -v`; `-v` explicitly deletes it.

## Log and shutdown review

Capture logs without terminal decoration and inspect the saved artifact. Normal
request telemetry includes operation, outcome, duration, and correlation ID; it
must not include request bodies, source text, generated answers, or credentials.

```bash
docker compose logs --no-color > /tmp/rag-kit-e2e.log
docker compose down
```

`docker compose down` preserves the named volume. Use `docker compose down -v`
only when you explicitly intend to delete this Compose project's index state.
Never point a cleanup command at an unverified Docker volume name or broad host
path.

## Operator changes

- Replace the two read-only bind-mount sources in `compose.yaml` to use another
  profile or corpus. Keep the container destinations unchanged unless the TOML
  paths change with them.
- Set `RAGKIT_PORT` to change only the loopback host port, for example
  `RAGKIT_PORT=8080 docker compose up -d --wait`.
- Override the service command only with the documented `ragkit-http` launcher;
  the entrypoint refuses missing config, corpus, state, or command prerequisites
  without printing environment values.
- Back up `/var/lib/ragkit` using volume-aware tooling while the service is
  stopped. Test restores against the same image and profile fingerprints.
