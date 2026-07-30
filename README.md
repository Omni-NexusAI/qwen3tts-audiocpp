# qwen3tts-audiocpp

One self-contained CUDA audio.cpp Qwen3-TTS container with the orange Gradio
Voice Studio and a preserved HF Realtime test page. It includes unquantized
Qwen3-TTS 0.6B and 1.7B Base checkpoints, targets broad local CUDA hardware,
and is intended to be faster than heavier backend options for realtime-oriented
testing. Performance comparisons are workload and hardware dependent.

`main` is the verified buffered/progressive phrase-PCM baseline. The separate
`development` branch is reserved for experimental native PCM engine work; it
must not be described as available until a single request emits real ordered
incremental PCM chunks.

## Boundary

- The endpoint is `http://127.0.0.1:8890/v1`, separate from Faster on `8881`
  and the user-managed Groxaxo candidate on `8882`.
- Both original BF16 Qwen checkpoints are staged in the candidate image during
  build. The only named volume is private persistent Voice Studio clone data.
- The Compose profile is opt-in. This repository never starts it automatically.
- The container supervisor keeps exactly one model process/configuration active.
  Switching models terminates the old process before starting the new one; both
  checkpoint files remain in the same image.
- Before a switch or synthesis, the supervisor checks free VRAM and GPU
  utilization. A busy GPU produces a clear `GPU busy/insufficient VRAM` result
  without attempting the operation. Load and synthesis reserves are reported
  separately, and named lifecycle events are available through the model-status
  endpoint and container logs.
- The Studio exposes a candidate-only **GPU admission guard**. Enforced mode
  uses 5,500 MiB free for 0.6B or 10,500 MiB for 1.7B and 85% load utilization;
  synthesis defaults to a separate 2,048 MiB / 95% check. Custom mode persists
  user-selected coexistence reserves in the private voice volume. Disabled mode
  explicitly bypasses preflight only and warns that CUDA OOM remains possible.
- Do not substitute a Q8 or other reduced-weight checkpoint for this initial
  candidate.

## Bring-up and validation

1. Start it explicitly from the repository root. The build downloads only the
   two required original BF16 Base models and their tokenizer sidecars:

   ```powershell
   docker compose up --build
   ```

2. Open `http://127.0.0.1:8891/voice-studio/` for the candidate's copied
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

The Studio stores Base clone metadata and WAVs in its private volume. Its
candidate adapter resolves `clone:<profile_id>` to that private audio and sends
audio.cpp the documented `voice_ref` and `reference_text` fields. The live
Playground surface can test microphone selection, browser VAD, pre-roll,
cancellation, llama.cpp turn flow, and progressive phrase PCM playback through
same-origin candidate proxies. audio.cpp Qwen3 has no native incremental PCM
mode, so each phrase completes before playback; this is explicitly labeled
buffered scaffolding and is not a realtime replacement or first-chunk benchmark.
The streaming Playground starts by default for evaluation. Non-streaming keeps audio.cpp's WAV as its quality master
and returns exactly one selected format: WAV and PCM stay native/lossless, FLAC
is lossless, and MP3/AAC/Opus use high-quality 320 kbps final encodes. The
result pane labels the one output's container separately from its codec (for
example, `WAV / PCM S16LE` is one WAV file, not two outputs).
Gradio's `-1` random-seed sentinel is removed by the adapter before synthesis;
non-negative explicit seeds are normalized to unsigned integer values for
audio.cpp.
