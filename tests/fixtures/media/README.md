# Media fixtures

These redistributable synthetic fixtures contain no real caller audio or video.
`support.wav` contains speech synthesized by FFmpeg's libflite filter, and
`training.mp4` contains two generated color scenes plus synthesized narration.
Opt-in modality tests decode them with the pinned faster-whisper and PySceneDetect
adapters; they prove execution and timestamp provenance, not production accuracy.
