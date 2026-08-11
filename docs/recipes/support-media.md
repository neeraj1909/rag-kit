# Timestamped support and training-media retrieval

Family: `media`

## Business use case

Search support calls and training clips, then jump from an answer to the cited audio
interval or video keyframe instead of reviewing an entire recording.

## Contract

`MediaDocumentExtractor` implements `DocumentExtractor`. Transcript and scene times
are finite, monotonic, non-overlapping half-open millisecond intervals; keyframes link
to their scene and original asset.

## Config schema

Use `configs/media.toml`: declared classifier, extractor `"media"`, evidence chunker.
`AdapterSettings` pins the ASR revision and bounds duration, segments, scenes, and
timeout; outer `RuntimeLimits` bounds bytes, documents, parts, and chunks.

## Registry and bootstrap

The `"media"` factory injects `LocalFasterWhisperTranscriber` and
`PySceneDetectBackend`. A replacement needs a distinct selection/factory and updated
capability inspection; video must retain both audio and visual processing.

## Tests

Run media fake/unit contracts plus opt-in ASR/scene integration. Assert timestamp
normalization, linked keyframes, cleanup on success/failure, silence/no-scene behavior,
and refusal of transcript-only video.

## Optional extra

Install `rag-kit[media]` and provision `Systran/faster-whisper-tiny.en` at
`0d3d19a32d3338f10357c0889762bd8d64bbdeba`; decoder availability remains an operator
prerequisite.

```bash
uv run --with huggingface-hub hf download \
  Systran/faster-whisper-tiny.en \
  --revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba
```

## Limits

The baseline caps media at 30 minutes, 200 transcript segments, and 100 scenes, with
one midpoint keyframe per scene. Profiles also bound bytes, timeout, documents, parts,
and chunks.

## Determinism

The pinned local CPU/int8 transcriber and configured scene threshold produce stable
normalized ordering for fixed artifacts. Model/dependency changes require a new
fingerprint and evaluation.

## Confidence and fallback

ASR confidence and speaker identity are unavailable in the baseline; segments carry
`speaker_unknown`. Video never falls back to transcript-only success when its visual
stream was not processed.

## Failure modes

Unsupported codec, decode/transcription/scene failure, invalid timestamps, silence or
no-scene cases, missing cache/dependency, and duration/segment/scene/time limits are
explicit. Isolated temporary files are removed after success or failure.
