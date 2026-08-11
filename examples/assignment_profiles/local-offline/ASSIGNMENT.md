# Local offline assignment

Use this template when the assessment forbids network or paid providers. It
combines hashing embeddings, in-memory dense search, BM25/RRF hybrid retrieval,
and extractive generation. The index is process-local and rebuilt by each CLI
invocation.

1. Put synthetic or approved source files under `data/`.
2. Run `uv run ragkit inspect-config --config ragkit.toml`.
3. Run `uv run ragkit ask --config ragkit.toml "<your question>"`.

Customize `ragkit.toml` first. Change toolkit internals only when a required
contract is genuinely missing, then follow the repository extension guide and
shared behavioral contracts.
