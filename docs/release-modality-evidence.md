# Release modality evidence

The five-family evaluation matrix remains the aggregate quality gate. The
video profile adds one real, opt-in application-path check that the audio-only
matrix row cannot cover.

## Video evidence path

`configs/media-video.toml` indexes the synthetic training video through the
normal composition root. The integration check asks which training step
requires closing the isolation valve. That wording follows the maintenance
training use case without embedding the expected answer.

The check proves two different evidence paths with separate user questions:

- the generated answer cites the retrieved transcript chunk and preserves its
  exact `TimeSpanLocator`;
- a reviewer-oriented scene-preview question retrieves only three of the five
  chunks, ranks a linked keyframe first, and cites provenance containing both
  its primary `KeyframeLocator` and related scene `TimeSpanLocator`;
- the relation-bearing chunks retain the original video asset and explicit
  `keyframe_of` relationship, and the cited keyframe ranks ahead of evidence
  from an unrelated scene.

Run the reviewed, locally provisioned model path without network access:

```bash
ATEN_CPU_CAPABILITY=default \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
RAGKIT_RUN_MODEL_INTEGRATION=1 \
uv run --frozen --all-extras pytest \
  tests/integration/test_phase6_modality_e2e.py -q
```

This is fixture-level execution evidence. It does not measure ASR accuracy on
real recordings, scene-boundary quality across camera styles, speaker identity,
or visual understanding of the selected keyframes. The media adapter reports
those unavailable or derived signals explicitly instead of manufacturing
confidence.

## Vision matrix observation

The vision matrix entry uses the reviewed, immutable SmolVLM revision and the
full-image `BoxLocator`. Release reports must only mark the row as passing when
the model actually executed and all recorded extraction, retrieval, citation,
and locator metrics passed. A capability preflight or an empty/incomplete
report is not execution evidence.
