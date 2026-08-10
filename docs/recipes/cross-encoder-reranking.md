# Local cross-encoder reranking

`LocalCrossEncoderReranker` uses
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a)
at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. The official model card
identifies it as an English MS MARCO text-ranking cross-encoder under the
Apache-2.0 license. That is a relevance model, not a factuality or answer-quality
judge.

Provision the reviewed revision explicitly before starting an offline process:

```bash
hf download cross-encoder/ms-marco-MiniLM-L6-v2 \
  --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

Install the `reranking` project extra selected by the profile. Runtime loading
uses `local_files_only=True` and `trust_remote_code=False`; a missing dependency
or cache entry fails with an actionable error instead of downloading weights.
Inference is CPU-only, puts the model in evaluation mode, uses PyTorch inference
mode, truncates pairs at the configured maximum sequence length, and processes
only bounded candidate batches.

To run the real smoke test after provisioning:

```bash
RAGKIT_RUN_MODEL_INTEGRATION=1 \
  uv run pytest tests/integration/test_cross_encoder_reranker_integration.py --no-cov
```

The adapter preserves the exact input `Chunk`, records score history newest-to-oldest
(the incoming retrieval score followed by its older history), and emits the model's
finite raw logit as higher-is-better relevance. Equal logits are ordered by stable
chunk ID. Compare task-specific quality through the Phase 4 evaluation harness; the
upstream model-card benchmark is not evidence for a local corpus.
