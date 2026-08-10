# Timestamped support and training-media retrieval

Use `MediaDocumentExtractor` with an explicitly provisioned transcriber for audio.
Video also requires a scene detector; transcript-only processing of video fails as
partial extraction rather than hiding the unprocessed visual stream.

Provider transcript seconds are normalized to half-open millisecond intervals.
Segments must be finite, monotonic, non-overlapping, and bounded by the declared
media duration. Video scenes obey the same ordering and duration bounds, while each
representative keyframe points back to its scene and original asset. The reference
limits are 30 minutes, 200 transcript segments, and 100 scenes.

The baseline does not perform diarization. Every transcript segment carries a
`speaker_unknown` notice, and no name or speaker label is inferred from turn order,
voice, filename, or script. ASR correctness confidence is unavailable unless a
future engine contract exposes and documents a raw signal.

The committed media files are redistributable FFmpeg-generated fixtures: a
libflite speech WAV and a two-scene narrated MP4. Opt-in tests exercise the real
faster-whisper and PySceneDetect adapters. They prove decoding, normalized
timestamps, linked keyframes, and provenance—not production accuracy.

The concrete offline ASR baseline is `Systran/faster-whisper-tiny.en` at
`0d3d19a32d3338f10357c0889762bd8d64bbdeba`. Provisioning is an explicit
operator step:

```bash
uv run --with huggingface-hub hf download \
  Systran/faster-whisper-tiny.en \
  --revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba
```

`LocalFasterWhisperTranscriber` uses CPU/int8 and `local_files_only=True`;
`PySceneDetectBackend` uses the configured content threshold and deterministic
scene-midpoint keyframes. Both decode through isolated temporary files which are
removed after success or failure. Release evidence provisions this exact cache
explicitly and runs with network access disabled.
