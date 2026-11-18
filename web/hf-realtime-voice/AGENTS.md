# HF Realtime Candidate Client

## Purpose

- Owns the bundled browser client used to exercise the standalone candidate through same-origin status, profile, clone, speech, and Voice Studio routes.

## Local Contracts

- Keep this directory frontend-focused; engine and model lifecycle behavior belongs to the audio.cpp supervisor.
- Scope voice inventory and tuning to the selected provider. Never expose stale clone IDs or send audio.cpp tuning to another provider.
- Candidate profile REST requests use `scope: "realtime"`; session WebSocket `tts_tuning` carries only provider, profile ID, overrides, and resolved runtime values.
- Startup is fail-fast: wait for the pipeline configuration acknowledgement before entering Listening or uploading microphone frames, and surface pre-ack errors or timeout instead of looping on Connecting.
- Treat native PCM as available only when live capability and response headers prove it. Label buffered and offline delivery accurately.
- Native browser AEC remains the default. Adaptive uses only the bundled hash-verified AEC3 module and must fall back truthfully; physical loopback and human double-talk remain user validation.
- Keep API keys browser-session-only and persist only non-secret UI selections through server-managed settings.

## Verification

- Run `node --check` on changed JavaScript modules, Python syntax checks on the server, focused UI/proxy tests, and `node integrations/audio-cpp/tests/realtime_config_ack.test.mjs` for the fail-closed startup contract. Run the Studio playback test and AEC3 smoke test when applicable.
