# Hosted generation with persistent retrieval

Use this template when the assessment permits a hosted answer model and needs a
local index that survives process restarts. Chroma stores the index under
`.ragkit/chroma`; OpenAI credentials are read from `OPENAI_API_KEY` only at
composition and are never stored in this profile.

1. Install `uv sync --frozen --group dev --extra persistent --extra hosted`.
2. Put synthetic or approved source files under `data/` and export the credential.
3. Run `uv run ragkit inspect-config --config ragkit.toml` before indexing or asking.

Do not commit credentials or `.ragkit/`. Customize `ragkit.toml` first; use the
repository extension guide and shared contracts only when a new adapter is
required.
