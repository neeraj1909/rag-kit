# Local cross-encoder reranking

## Business use case

Rerank a bounded text candidate set when first-stage retrieval returns several
lexically plausible policy passages and the most query-relevant evidence should appear
first. This does not judge answer correctness.

## Contract

`LocalCrossEncoderReranker` implements `Reranker`: it returns a subset only, retains
exact chunks and prior scores, emits finite raw logits as higher-is-better relevance,
and uses stable chunk-ID tie breaks.

## Config schema

Use `configs/reranked.toml`. Select `reranker = "cross-encoder"`; model ID/revision,
batch size, sequence length, candidate count, and top-k are validated `AdapterSettings`
fields.

## Registry and bootstrap

The `"cross-encoder"` factory in `infrastructure/bootstrap.py` constructs the adapter
from validated settings. A different model policy needs an explicit selection/factory
rather than changing this pinned selection in place.

## Tests

Unit tests use an injected backend and the shared reranker contract. After provisioning
the exact cache, run the real CPU check:

```bash
RAGKIT_RUN_MODEL_INTEGRATION=1 \
  uv run pytest tests/integration/test_cross_encoder_reranker_integration.py --no-cov
```

Compare task quality through the Phase 4 evaluation runner; the upstream model-card
benchmark is not evidence for a local corpus.

## Optional extra

Install `rag-kit[reranking]`. The baseline is the Apache-2.0 English MS MARCO model
`cross-encoder/ms-marco-MiniLM-L6-v2` at immutable revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Provision it outside the request path:

```bash
hf download cross-encoder/ms-marco-MiniLM-L6-v2 \
  --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

## Limits

Candidate count, output top-k, batch size, and tokenized sequence length are bounded;
input pairs are truncated only under the configured sequence-length policy.

## Determinism

CPU-only, local-files-only inference uses `trust_remote_code=False`, the pinned
revision, evaluation mode, and PyTorch inference mode. Fixed dependencies, inputs, and
settings produce stable order.

## Confidence and fallback

The raw logit is a ranking score, not probability, factuality, or confidence. Missing
model capability has no no-op fallback.

## Failure modes

Missing dependency/cache, invalid or duplicate candidates, non-finite backend scores,
or configured bounds fail explicitly without introducing or silently dropping
evidence.
