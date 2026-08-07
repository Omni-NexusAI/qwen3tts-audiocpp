# audio.cpp Qwen3-TTS Candidate

This is one self-contained CUDA candidate container for audio.cpp plus the
completed copied Voice Studio. The image includes unquantized Qwen3-TTS 0.6B
Base and 1.7B Base checkpoints. It does not change, mount, rebuild, or manage
the `qwen3-tts-faster` container.

## Boundary

- The endpoint is `http://127.0.0.1:8890/v1`, separate from Faster on `8881`
  and the user-managed Groxaxo candidate on `8882`.
- Both original BF16 Qwen checkpoints are staged in the candidate image during
  build. The only named volume is private persistent Voice Studio clone data.
- The Compose profile is opt-in. This repository never starts it automatically.
- The regular candidate file pins `AUDIO_CPP_NATIVE_INCREMENTAL_PCM=false`.
  Native development is enabled only by adding `compose.native.yml`; buffered
  phrase PCM and complete WAV remain available in that same candidate. The
  override only changes the runtime flag on the exact image selected through
  `AUDIO_CPP_IMAGE`; the current tag is `native-development`, and the sole
  retained recovery image is `rollback-pre-native-profile-fix-20260805`.
- The native override disables per-shape CUDA graph capture because changing
  conversational prompt shapes otherwise reintroduces a multi-second first-block
  capture. Explicit **Load** therefore includes one private, discarded native
  warmup using a live paired clone before the model reports `loaded`. Status
  exposes `warming`, elapsed time, profile ID, timeout, and retryable
  skipped/failed results; the warmup never appears as a user generation metric.
- The container supervisor keeps exactly one model process/configuration active.
  Switching models terminates the old process before starting the new one; both
  checkpoint files remain in the same image.
- Before a switch or synthesis, the supervisor checks free VRAM and GPU
  utilization. A busy GPU produces a clear `GPU busy/insufficient VRAM` result
  without attempting the operation. Load and synthesis reserves are reported
  separately, and named lifecycle events are available through the model-status
  endpoint and container logs.
- The Studio exposes a candidate-only **GPU admission guard**, defaulting to
  **Disabled** so the expected chat LLM plus one audio.cpp model can coexist.
  Enforced and Custom remain explicit opt-ins. For 0.6B,
  Enforced admission uses the observed ~5,630 MiB load delta rounded up to a
  6,000 MiB model/graph reserve plus the unchanged 2,048 MiB synthesis floor:
  8,048 MiB free before load at 85% maximum utilization. For 1.7B, Enforced
  keeps the existing 10,500 MiB **total** threshold until a real residency
  delta is measured; it does not invent and add a second 2,048 MiB reserve.
  Synthesis retains its separate 2,048 MiB / 95% check.
  Custom mode uses its persisted load value as an explicit absolute threshold;
  Disabled mode bypasses preflight only and warns that CUDA OOM remains possible.
- Do not substitute a Q8 or other reduced-weight checkpoint for this initial
  candidate.

## Bring-up and validation

1. A first bootstrap, when no validated local base image exists yet, may use
   the full Dockerfile. It downloads the two original BF16 Base models and
   their tokenizer sidecars, compiles the pinned engine, and creates the
   self-contained candidate image:

   ```powershell
   docker build -f integrations/audio-cpp/Dockerfile -t local/audio-cpp-qwen3-tts-voice-studio:native-development .
   ```

2. A repaired live rollout must use `Dockerfile.overlay` explicitly. It
   compiles the repaired pinned audio.cpp binary in a disposable build stage,
   then copies that binary plus the current supervisor and UI sources onto the
   validated model-bearing base image. It does not re-download either BF16
   checkpoint:

   ```powershell
   docker build -f integrations/audio-cpp/Dockerfile.overlay -t local/audio-cpp-qwen3-tts-voice-studio:native-development .
   $env:AUDIO_CPP_IMAGE = 'local/audio-cpp-qwen3-tts-voice-studio:native-development'
   docker compose -p audio-cpp -f integrations/audio-cpp/compose.candidate.yml -f integrations/audio-cpp/compose.native.yml --profile audio-cpp-candidate up -d --no-build --no-deps --force-recreate audio-cpp-candidate
   Remove-Item Env:AUDIO_CPP_IMAGE
   ```

   Keep project name `audio-cpp`; it preserves the existing container and
   private-volume identity. Never run `docker compose down -v` during rollout
   or rollback because `-v` would remove the persisted clone/profile library.
   `AUDIO_CPP_IMAGE` selects the exact existing image consumed by Compose, so a
   rollback does not trigger a build:

   ```powershell
   $env:AUDIO_CPP_IMAGE = 'local/audio-cpp-qwen3-tts-voice-studio:rollback-pre-native-profile-fix-20260805'
   docker compose -p audio-cpp -f integrations/audio-cpp/compose.candidate.yml --profile audio-cpp-candidate up -d --no-build --no-deps --force-recreate audio-cpp-candidate
   Remove-Item Env:AUDIO_CPP_IMAGE
   ```

3. Open `http://127.0.0.1:8891/voice-studio/` for the candidate's copied
   Gradio Voice
   Studio. It is the orange copied Gradio surface with visible candidate model
   controls. The root page remains the separate HF Realtime integration test UI.
   Confirm the
   private API endpoint `http://127.0.0.1:8890` reports both
   `qwen3-tts-0.6b-base-bf16` and `qwen3-tts-1.7b-base-bf16`.
4. Select a candidate model in **Settings & candidate model controls**, click
   **Load selected model**, and use the Base clone or Playground to run a
   buffered speech probe. **Unload model** terminates the audio.cpp child and
   leaves no model resident.

The image uses a multi-stage CUDA build: compilation happens in a development
stage, while the final image contains only the server binary, required runtime
assets, Python/Gradio/FFmpeg runtime, and model files. It excludes compiler
tooling, CUDA development packages, Nsight tooling, build trees, source/Git
metadata, and tests. Both model directories remain original BF16 Hugging Face
checkpoints; their identical speech-tokenizer weights are stored once through
internal symlinks. This reduces storage without changing checkpoint weights or
runtime math.

The Studio stores Base clone metadata and WAVs in its private volume. Gradio
and the API rescan the same live directories: hidden/cache entries, missing
reference audio, malformed metadata, and non-Base entries are excluded. Safe
legacy Base metadata is normalized atomically without changing its directory
ID or reference recording. Its
candidate adapter resolves `clone:<profile_id>` to that private audio and sends
audio.cpp the documented `voice_ref` and `reference_text` fields. The live
adapter treats those fields as one conditioning pair: it never shortens a WAV
while retaining the full transcript. `max_reference_seconds` can select a
shorter reference only when profile metadata stores that excerpt audio with its
exact excerpt transcript; otherwise the complete pair is used and response
headers report that the requested limit was not applied.
The live
Playground surface can test microphone selection, browser VAD, pre-roll,
cancellation, llama.cpp turn flow, and progressive phrase PCM playback through
same-origin candidate proxies. The normal candidate stays buffered: each phrase
completes before playback and is not labeled native streaming. The development
override enables the pinned eight-file engine patch, keeps one model resident,
and relays model-incremental 24 kHz PCM16 chunks through `/v1/audio/speech` with
`stream:true` and `response_format:"pcm"`. Stop/new-turn cancellation closes the
browser proxy and supervisor streams so the engine resets without stale tail
playback. When that native runtime is enabled, Native incremental PCM is the
Playground default; buffered phrase PCM and full WAV stay selectable. This
native mode is live-validated independently with both raw models for
multi-chunk delivery, cancellation/recovery, timing, and single-resident GPU
behavior; select buffered phrase PCM for immediate rollback. Physical
speaker-loopback AEC3 suppression and human double-talk preservation remain
user-owned acoustic validation.
The streaming Playground starts by default for evaluation. Non-streaming uses
one complete offline decode and keeps audio.cpp's PCM16 WAV as its full-quality
master. It always uses the shipped Quality sampler policy independently of
streaming selections or session overrides; first/steady/context and phrase
queue controls are disabled while Full WAV is selected. The result pane labels
this path `offline-full-decoder`. The supervisor requests
`qwen3_tts.decode_mode=offline_full`, rejects a contradictory result-derived
decoder-mode header, and never retries through incremental generation. WAV is
returned unchanged from that completed master, raw PCM is extracted directly,
 FLAC is lossless, MP3 and AAC are post-encoded at 320 kbps, and Opus uses the
 codec's maximum supported 256 kbps without resampling or downmixing.
 `/v1/audio/speech` with `stream:false` plus PCM remains
the separately labeled buffered fallback; `/v1/audio/voice-clone` is always the
full offline policy regardless of export format.
Native and buffered streaming use the selected Voice Studio profile directly.
Punctuation-complete text dispatches immediately; incomplete text uses the
profile look-ahead and idle-flush values, cuts only on safe word/clause
boundaries, and has a profile-derived hard cap. Browser-local legacy phrase
values never override the selected profile. One turn-scoped AudioWorklet queue
spans every phrase, waits for the first-plus-steady native duration before its
initial start, reports underruns/drain, and rejects late PCM after cancellation.
Non-secret Studio controls (LLM endpoint/model/prompt, selected microphone,
VAD, and selected tuning identity) are atomically retained in the private volume, so they
survive candidate recreation. API keys intentionally remain browser-only. A
bare local llama.cpp server address such as `http://127.0.0.1:8818` is accepted
and normalized to `/v1/chat/completions`; a full OpenAI-compatible endpoint is
also accepted. The Streaming Playground captures mono Float32 PCM through an
AudioWorklet, applies VAD and pre-roll over PCM buffers, and creates a complete
PCM16 RIFF/WAV for every accepted turn. Lossless WAV at 16 kHz is the default;
24 kHz and 48 kHz encodes and an explicit server-converted MP3 320 kbps A/B
transport are available. llama.cpp input metadata always matches the bytes
sent. A candidate restart always releases the active model process, so
the Studio must explicitly load its selected model before its next generation.
Quality, Balanced, Low Latency, and custom tuning definitions are shared, but
Voice Studio and Realtime persist independent active selections. Balanced is
the fallback for either surface when it has no valid saved selection.
Built-ins are immutable; editing starts by cloning one. Realtime Save As sends
the complete effective safe schema and can atomically select the new profile
only for the Realtime scope, without changing Voice Studio's selection. Custom
updates require their current revision and reject stale writes.
All built-ins retain the model-required 72 decoder-context frames. A smaller
custom or temporary value is an explicitly warned Advanced experiment because
reduced causal history can progressively degrade a long native stream. The
pinned native engine patch uses the same 72-frame fallback for direct internal
requests, so bypassing profile resolution cannot silently restore the older
quality-degrading context.
Seed accepts null or an unsigned 32-bit integer: blank and Gradio's `-1`
sentinel omit the field for engine randomness, while explicit values are
forwarded. Temperature must be greater than zero. x-vector-only clone mode and
crossfade remain inactive. Full ICL clone reference audio plus its transcript
remains required.
