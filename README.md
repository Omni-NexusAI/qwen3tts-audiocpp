# qwen3tts-audiocpp

Run Qwen3-TTS voice cloning locally through a CUDA-accelerated audio.cpp
backend and a complete browser-based Voice Studio. This project is for people
who want to create and manage reusable voices, compare the raw 0.6B and 1.7B
Base models, export finished audio, or evaluate low-latency speech without
assembling the engine, model files, API, and interface themselves.

Everything runs in one isolated container: the audio.cpp engine, both original
BF16 Base checkpoints, the orange Gradio Voice Studio, and a Hugging Face
Realtime integration test page. Only one model is loaded on the GPU at a time.

> **Development status:** full-quality offline synthesis and buffered phrase
> playback remain the dependable paths. Native model-incremental PCM is an
> opt-in experimental mode with a buffered rollback path. The included AEC3
> echo-control work still needs physical speaker-loopback and human double-talk
> validation before it should be treated as production-ready.

## What you can do

- Create Base voice clones from a reference WAV and its exact transcript.
- Edit, rename, select, import, export, and delete persistent voice profiles.
- Switch between the original, unquantized Qwen3-TTS 0.6B and 1.7B Base
  checkpoints without loading both at once.
- Generate full-quality WAV, raw PCM, FLAC, MP3, AAC, or Opus output from one
  completed offline decode.
- Compare Quality, Balanced, Low Latency, and custom synthesis profiles while
  keeping Voice Studio and Realtime selections independent.
- Evaluate native 24 kHz PCM16 blocks as the model generates them, or return to
  buffered phrase PCM without changing checkpoints.
- Exercise microphone capture, browser VAD, cancellation, llama.cpp turn flow,
  continuous PCM playback, and echo-control diagnostics from the bundled test
  surfaces.
- Use a loopback-only OpenAI-compatible speech API alongside model, voice,
  profile, tuning, health, and lifecycle endpoints.

## Is it a fit?

This repository is a good fit when you want local Qwen3-TTS Base cloning on an
NVIDIA GPU and are comfortable running Docker. The documented quick path is
tested with Windows, PowerShell, Docker Desktop, and GPU-enabled Linux
containers.

You need:

- an NVIDIA GPU and driver configuration that Docker can access;
- Docker Desktop with Docker Compose;
- enough local storage and download bandwidth for both raw BF16 checkpoints,
  CUDA layers, and build layers; and
- enough GPU headroom for your chosen TTS model plus anything else you run at
  the same time.

The project does not ship quantized checkpoints. Memory use, first-audio time,
and real-time factor vary by GPU, model, reference, text, and concurrent GPU
work. The 0.6B model is the lighter comparison point; the 1.7B model generally
requires more headroom. A single-resident supervisor always stops the previous
audio.cpp model before starting the other one.

## Quick start on Windows

The first build compiles the pinned audio.cpp engine and downloads both Qwen3
Base checkpoints. Run these commands in PowerShell:

```powershell
git clone --branch development https://github.com/Omni-NexusAI/qwen3tts-audiocpp.git
Set-Location qwen3tts-audiocpp

docker build `
  -f integrations/audio-cpp/Dockerfile `
  -t local/audio-cpp-qwen3-tts-voice-studio:native-development `
  .

$env:AUDIO_CPP_IMAGE = 'local/audio-cpp-qwen3-tts-voice-studio:native-development'
docker compose `
  -p audio-cpp `
  -f integrations/audio-cpp/compose.candidate.yml `
  -f integrations/audio-cpp/compose.native.yml `
  --profile audio-cpp-candidate `
  up -d --no-build audio-cpp-candidate
Remove-Item Env:AUDIO_CPP_IMAGE
```

Open [http://127.0.0.1:8891/voice-studio/](http://127.0.0.1:8891/voice-studio/),
choose a model under **Settings & candidate model controls**, and click
**Load selected model**. Loading is explicit: starting or recreating the
container does not leave a model resident automatically.

You can confirm the service before generating audio:

```powershell
Invoke-RestMethod http://127.0.0.1:8890/health
Invoke-RestMethod http://127.0.0.1:8890/v1/backend/models
```

The quick start above enables experimental native PCM because it is the
lowest-latency supported configuration. To begin with the dependable
buffered/offline runtime instead, omit the native override:

```powershell
$env:AUDIO_CPP_IMAGE = 'local/audio-cpp-qwen3-tts-voice-studio:native-development'
docker compose `
  -p audio-cpp `
  -f integrations/audio-cpp/compose.candidate.yml `
  --profile audio-cpp-candidate `
  up -d --no-build --no-deps --force-recreate audio-cpp-candidate
Remove-Item Env:AUDIO_CPP_IMAGE
```

The named voice-library volume survives this recreation. Do not use
`docker compose down -v`: `-v` removes the persisted clone, profile, and Studio
settings library.

## Using Voice Studio

### 1. Load one model

Use the model controls to load either:

- `qwen3-tts-0.6b-base-bf16`; or
- `qwen3-tts-1.7b-base-bf16`.

Switching terminates the old engine process before the new process starts.
**Unload model** releases the TTS model completely. The optional GPU admission
guard can enforce or customize preflight headroom; its default is Disabled so
one chat LLM and one audio.cpp model can intentionally coexist. The
single-resident rule is enforced in every guard mode.

### 2. Create a voice clone

Open **Create > Voice Clone (Base)**, provide a reference recording and the
exact words spoken in that recording, then save the profile. Audio and
transcript are treated as one conditioning pair. A reference-duration limit is
used only when the profile contains a shorter audio excerpt with its own exact
excerpt transcript; otherwise synthesis keeps the complete pair.

CustomVoice preset checkpoints and VoiceDesign checkpoints are not included.
Their tabs remain visible but unavailable so the interface does not imply
support that the Base-only backend does not provide.

### 3. Choose an output path

Voice Studio keeps offline and streaming behavior separate:

| Mode | Behavior | Output |
| --- | --- | --- |
| **Non-streaming (Full Quality)** | One complete offline decoder pass using the dedicated Quality sampler policy | WAV, PCM, FLAC, MP3, AAC, or Opus |
| **Native incremental PCM** | Experimental model-level chunks relayed and played as they arrive | Raw 24 kHz PCM16 |
| **Buffered phrase PCM** | Completes each phrase before PCM playback; retained as the streaming rollback | Raw 24 kHz PCM16 |

Full Quality never inherits streaming block sizes, phrase timing, or temporary
streaming overrides. Its completed 24 kHz PCM16 WAV is the master: WAV is
returned unchanged, PCM is extracted without re-encoding, FLAC is lossless,
MP3 and AAC use 320 kbps final encodes, and Opus uses 256 kbps without
resampling or downmixing.

Native playback uses one continuous AudioWorklet queue across the assistant
turn, waits for an initial two-block buffer, and rejects late audio after
cancellation. If native mode is not enabled at runtime, the interface labels
the result as buffered fallback rather than calling it native streaming.

### 4. Tune quality and latency

The built-in **Quality**, **Balanced**, and **Low Latency** profiles are
versioned defaults. Clone a built-in before editing it. Voice Studio and the
Realtime test page share the profile definitions but persist separate active
selections, so changing one surface does not silently change the other.

Profiles expose only runtime-safe controls:

- active raw model requirement;
- maximum reference duration, effective only for a matched excerpt pair;
- first and steady native PCM block sizes;
- text look-ahead and safe phrase-flush timing;
- temperature, top-k, top-p, repetition penalty, and optional uint32 seed; and
- decoder context, which should remain at the model-required 72 frames.

Reducing decoder context is an explicitly warned experiment because it can
degrade a long native stream. Output transport stays fixed at model-native
24 kHz PCM16. x-vector-only cloning and runtime crossfade are unavailable and
are not forwarded to audio.cpp.

## Realtime test page

The container root at [http://127.0.0.1:8891/](http://127.0.0.1:8891/) is a
separate Hugging Face Realtime integration surface. It can connect microphone
capture and a local OpenAI-compatible multimodal LLM endpoint to the audio.cpp
candidate, then show phrase, TTS, playback, reference-pairing, GPU, and echo
diagnostics.

This page is for integration testing, not a claim that browser acoustic echo
cancellation is finished. It bundles a WebRTC AEC3 AudioWorklet/WASM path, but
silence-at-mic speaker playback, physical device delay, and simultaneous human
speech still require real speaker-loopback validation on each device pair.

## API

The API listens only on `http://127.0.0.1:8890`; its OpenAI-compatible base URL
is `http://127.0.0.1:8890/v1`.

Useful endpoints include:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Runtime capabilities, active model, GPU policy, and native-mode status |
| `GET /v1/models` | Both bundled model IDs and current selection |
| `GET /v1/backend/models` | Model lifecycle and single-resident status |
| `POST /v1/backend/models/switch` | Load or switch the active model |
| `POST /v1/backend/models/unload` | Release the current model |
| `GET /v1/voices` | Live candidate-local Base clones |
| `GET /v1/voices/profiles` | Editable voice-profile inventory |
| `GET /v1/tuning/profiles` | Built-in and custom tuning definitions |
| `POST /v1/tuning/resolve` | Resolve effective profile behavior and inactive fields |
| `POST /v1/audio/speech` | OpenAI-style speech synthesis and native/buffered PCM |
| `POST /v1/audio/voice-clone` | Full-quality offline clone synthesis in all six formats |

After replacing `PROFILE_ID` with a live clone ID and loading the selected
model, this PowerShell request writes a complete full-quality WAV:

```powershell
$body = @{
  model = 'qwen3-tts-1.7b-base-bf16'
  input = 'The local voice service is ready.'
  voice = 'clone:PROFILE_ID'
  response_format = 'wav'
  stream = $false
} | ConvertTo-Json

Invoke-WebRequest `
  -Uri http://127.0.0.1:8890/v1/audio/speech `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body `
  -OutFile output.wav
```

For native delivery, send `stream: true` with `response_format: "pcm"` to
`/v1/audio/speech`. The response is headerless signed 16-bit little-endian mono
PCM at 24 kHz, so the client must consume chunks incrementally and supply those
format parameters during playback. `stream: false` plus `response_format: "pcm"`
is the separately labeled buffered fallback.

Reference-pairing headers report source duration, requested limit, used
duration, whether the limit was applied, pairing type, and delivery mode. They
never include the transcript content.

## Known limitations

- Only Qwen3-TTS Base full-ICL voice cloning is enabled. CustomVoice,
  VoiceDesign, and x-vector-only clone modes are unavailable.
- Native PCM is experimental and requires the explicit Compose override.
- Runtime crossfade is fixed at zero; decoder continuity comes from the
  model-required causal context and complete codec-frame boundaries.
- AEC3 physical speaker-loopback suppression and double-talk preservation are
  not yet validated across real microphone/output-device pairs.
- The initial self-contained build is large because it compiles CUDA code and
  embeds both unquantized checkpoints.
- GPU guard thresholds are admission checks, not guarantees against contention
  from other applications after synthesis starts.
- Performance comparisons against other TTS backends depend on hardware and
  workload; this project does not promise a universal speedup.

## Operations and rollback

The container is intentionally separate from FasterQwen3TTS and Groxaxo. It
binds only loopback ports `8890` and `8891`, uses the Compose project name
`audio-cpp`, and stores user state in the named
`speech-to-speech-audio-cpp-voice-library` volume.

Inspect it with:

```powershell
docker compose `
  -p audio-cpp `
  -f integrations/audio-cpp/compose.candidate.yml `
  --profile audio-cpp-candidate `
  ps

docker logs audio-cpp-qwen3-tts-voice-studio
```

To return from native mode to the buffered runtime while keeping the same
image and data, recreate through the regular candidate file only:

```powershell
$env:AUDIO_CPP_IMAGE = 'local/audio-cpp-qwen3-tts-voice-studio:native-development'
docker compose `
  -p audio-cpp `
  -f integrations/audio-cpp/compose.candidate.yml `
  --profile audio-cpp-candidate `
  up -d --no-build --no-deps --force-recreate audio-cpp-candidate
Remove-Item Env:AUDIO_CPP_IMAGE
```

For a retained pre-native recovery image, select the rollback tag explicitly:

```powershell
$env:AUDIO_CPP_IMAGE = 'local/audio-cpp-qwen3-tts-voice-studio:rollback-pre-native-profile-fix-20260805'
docker compose `
  -p audio-cpp `
  -f integrations/audio-cpp/compose.candidate.yml `
  --profile audio-cpp-candidate `
  up -d --no-build --no-deps --force-recreate audio-cpp-candidate
Remove-Item Env:AUDIO_CPP_IMAGE
```

That rollback tag is an operational recovery convention, not an artifact a
fresh clone downloads automatically. Keep at most one deliberate rollback and
never remove an image that a container references.

Existing deployments can use `integrations/audio-cpp/Dockerfile.overlay` to
compile the current pinned engine and copy only the binary and control-plane
sources onto a validated model-bearing base image. The overlay avoids
downloading the checkpoints again; it is not the first-install path.

## Development

The image pins audio.cpp commit
`238ab6a9e321c17de8e120559f57efeedaeb1345` and applies one reviewable Qwen3
native-PCM patch. A multi-stage CUDA build keeps compiler tooling and source
metadata out of the runtime image. Identical tokenizer weights shared by the
two model directories are stored once through internal symlinks without
changing checkpoint data or runtime math.

The main implementation boundaries are:

- `integrations/audio-cpp/`: Docker packaging, engine patch, single-resident
  supervisor, Voice Studio, profile persistence, and contract tests;
- `web/hf-realtime-voice/`: the bundled Realtime test surface and browser audio
  worklets; and
- the named `/voices` volume: persistent clones, tuning definitions, and
  non-secret Studio settings. API keys remain browser-session-only.

Representative checks are:

```powershell
python -m py_compile `
  integrations/audio-cpp/supervisor.py `
  integrations/audio-cpp/profile_library.py `
  integrations/audio-cpp/gradio_voice_studio.py `
  web/hf-realtime-voice/server.py

python -m pytest integrations/audio-cpp/tests
node --check web/hf-realtime-voice/main.js
node --test integrations/audio-cpp/tests/studio_playback.test.mjs
```

Keep engine changes pinned and patch-parity tested. Validate 0.6B and 1.7B
sequentially, preserve the buffered rollback, and do not load both models at
once.

## License

Repository code is licensed under the [Apache License 2.0](LICENSE). The image
also incorporates audio.cpp, Qwen model assets, FFmpeg, CUDA runtime
components, and the bundled AEC3 module; review their upstream licenses and
model terms for your distribution and deployment. The AEC3 source bundle keeps
its third-party license alongside the worklet assets.
