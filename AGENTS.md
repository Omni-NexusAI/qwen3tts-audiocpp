# Repository Contract

## Purpose

- This repository packages a local CUDA Qwen3-TTS Voice Studio around audio.cpp, including an isolated candidate API and an optional HF Realtime test client.
- The raw BF16 0.6B and 1.7B Base checkpoints are the supported model variants. Never make both models resident at once.

## Working Rules

- Read this file and the nearest child `AGENTS.md` before editing; update the owning contract after meaningful behavior or workflow changes.
- Preserve user-owned clone/profile data, model assets, unrelated containers, and the FasterQwen3TTS and Groxaxo comparison services.
- Keep native streaming truthful and reversible. Buffered PCM and complete offline output remain supported fallbacks.
- Use `native-development` as the canonical image tag; operators may select a deliberate rollback through `AUDIO_CPP_IMAGE`.
- Treat build products, caches, generated audio, model weights, and voice-library contents as runtime artifacts; do not commit them.
- Base feature branches on `development`, keep commits scoped, and use draft pull requests. Do not merge or publish a release without explicit authorization.

## Verification

- Run Python syntax checks, JavaScript syntax checks, focused integration tests, native patch parity, and the relevant browser/worklet smoke tests.
- Runtime claims require deployed health and behavior evidence; static tests alone do not prove CUDA, acoustic echo cancellation, or speaker similarity.

## Child DOX Index

- `integrations/audio-cpp/AGENTS.md` owns the engine, supervisor, Voice Studio, model lifecycle, profiles, and container contracts.
- `web/hf-realtime-voice/AGENTS.md` owns the bundled browser client and its same-origin candidate proxies.
