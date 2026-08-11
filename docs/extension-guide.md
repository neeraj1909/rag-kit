# Adapter extension guide

Add an adapter only when an existing selection cannot express the capability. Start
from the public contract, not from a provider SDK: read
`src/ragkit/ports/interfaces.py`, then its immutable requests and results in
`src/ragkit/ports/models.py`. The semantic rules summarized in
[`contracts.md`](contracts.md) remain authoritative for every implementation.

## Extension checklist

1. Choose one port and write down its ordering, effects, limits, score or confidence
   meaning, determinism, thread-safety, fallback, and failure behavior.
2. Add a narrowly named adapter under `src/ragkit/adapters/`. Keep provider values and
   imports behind that boundary; translate failures to typed `ragkit.domain` errors.
3. If configuration is needed, add typed fields and validation in
   `src/ragkit/infrastructure/config.py`. Do not accept an untyped provider-options bag
   or put secrets in a profile.
4. Add the selection and factory to the role-specific registry in
   `src/ragkit/infrastructure/bootstrap.py`. Extend capability inspection when the
   adapter needs a Python distribution, binary, credential, or provisioned model.
5. Put heavy dependencies in a named optional extra in `pyproject.toml`; core imports
   must still work without that extra.
6. Run the matching helper in `tests/contract_assertions.py` against the new adapter,
   then add unit tests for limits and errors and integration tests for the real
   dependency. A test fake is not evidence that a provider adapter works.
7. Add or update one recipe in `docs/recipes/` with the same checklist and a business
   use case. State unsupported inputs and degraded modes rather than implying support.

## Small guide-derived adapter

This deliberately small chunker demonstrates the complete shape of a dependency-free
adapter. It adds a configured prefix to every normalized part representation while
retaining the exact source locator. It is executable documentation: the contract test
extracts this block, constructs the adapter, and runs the shared chunker assertions.

```python guide-derived-adapter
from ragkit.domain import (
    Chunk,
    ChunkId,
    ComponentFingerprint,
    ImageContent,
    InvalidDomainValueError,
    LimitExceededError,
    MediaContent,
)
from ragkit.ports import Chunker, ChunkingRequest


class GuidePrefixChunker(Chunker):
    """Create one prefixed chunk per part without changing its evidence locator."""

    def __init__(self, prefix: str) -> None:
        if not prefix.strip():
            raise InvalidDomainValueError("prefix must not be blank")
        self._prefix = prefix.strip()
        self._fingerprint = ComponentFingerprint.create(
            "chunker", "guide_prefix", {"version": 1, "prefix": self._prefix}
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for document in request.documents:
            for ordinal, part in enumerate(document.parts):
                if isinstance(part, ImageContent):
                    representation = part.description
                elif isinstance(part, MediaContent):
                    representation = part.transcript
                else:
                    representation = part.text
                text = f"{self._prefix}: {representation}"
                chunks.append(
                    Chunk(
                        ChunkId.from_content(
                            document.document_id,
                            self.fingerprint,
                            ((part.part_id, part.provenance.locator),),
                            text,
                        ),
                        document.document_id,
                        ordinal,
                        text,
                        (part.provenance,),
                        (part.part_id,),
                    )
                )
        if len(chunks) > request.max_chunks:
            raise LimitExceededError(
                f"chunk limit {request.max_chunks} would truncate {len(chunks)} chunks"
            )
        return tuple(chunks)
```

The example is pure, deterministic, and thread-safe. It has no optional extra or
fallback. A blank prefix is invalid; an output that would exceed `max_chunks` fails
instead of truncating evidence.

## Configuration and composition example

For a production version, add `chunk_prefix: str` to `AdapterSettings`, validate it in
`AdapterSettings.__post_init__`, and add a `"prefix"` factory beside the other chunker
factories in `bootstrap`. A profile would select `chunker = "prefix"` and set
`chunk_prefix = "indexed"`. This is an explicit code registry change; profile strings
never cause dynamic imports.

If an adapter talks to a hosted service, store only the environment-variable name in
configuration and resolve the credential at composition time. Neither the secret nor a
hash of it belongs in fingerprints, logs, serialized profiles, or exceptions.

## Contract and test commands

Use the shared assertion matching the port and include positive and negative examples:

```bash
uv run pytest tests/contract/test_extension_documentation.py --no-cov
uv run pytest -m contract --no-cov
uv run ruff check .
uv run mypy src tests
```

Then run the adapter's unit and integration tests from `CONTRIBUTING.md`. Provisioned
models and live services remain explicit opt-ins; tests must not download weights or
make paid calls implicitly.

## Public docstring checklist

The port docstring is the behavioral contract. Adapter docstrings add only concrete
facts: supported formats or modes, ordering, limits, state changes and external I/O,
determinism conditions, thread safety, score or confidence meaning, fallback policy,
and typed failure behavior. Do not copy a provider marketing claim or describe an
uncalibrated model value as confidence.
