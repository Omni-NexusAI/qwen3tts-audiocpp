# audio.cpp Candidate

## Purpose

- Owns the pinned audio.cpp engine patch, one-container CUDA candidate, single-resident supervisor, private profile library, and adapted Gradio Voice Studio.

## Local Contracts

- Pin audio.cpp to the declared commit and require the packaged patch to apply cleanly to a fresh checkout. Do not silently follow an upstream branch.
- Keep only the original BF16 0.6B and 1.7B Base checkpoints. A model switch must stop the old engine before loading the new one.
- Keep clone reference audio and transcript indivisible. A duration limit applies only when a stored excerpt has its exact matching transcript.
- Full-quality non-streaming synthesis uses one complete offline PCM16/24 kHz master. Streaming settings and overrides must not affect this path.
- Native streaming emits incremental PCM16/24 kHz, keeps the required 72-frame decoder context by default, supports cancellation without stale tail audio, and retains buffered PCM as rollback.
- Voice Studio and Realtime share profile definitions but persist independent selections. Temporary overrides remain session-scoped.
- Preserve the named `/voices` volume and model-bearing image layers. Never use `down -v`, manage Faster/Groxaxo, or bypass the one-resident-model boundary.
- Unsupported CustomVoice, VoiceDesign, x-vector-only cloning, and adjustable transport bitrate remain visibly unavailable rather than simulated.

## Verification

- Run `python -m pytest integrations/audio-cpp/tests -q` with a writable temporary directory.
- Run `node integrations/audio-cpp/tests/studio_playback.test.mjs` and the AEC3 smoke test when browser audio assets change.
- Native promotion additionally requires multi-chunk headers, cancellation/recovery, offline fallback, and sequential live checks for both raw models.
