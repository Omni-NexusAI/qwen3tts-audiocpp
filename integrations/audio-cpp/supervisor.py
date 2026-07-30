"""Single-resident-model proxy for the self-contained audio.cpp candidate."""
from __future__ import annotations

import asyncio
import base64
from collections import deque
import io
import json
import logging
import os
import re
from urllib.parse import urlsplit, urlunsplit
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

TEMPLATE = Path(os.environ.get("AUDIO_CPP_CONFIG_TEMPLATE", "/config/qwen3-tts-base-f16.json"))
ACTIVE_CONFIG = Path("/run/audio-cpp-active.json")
ENGINE_BIN = os.environ.get("AUDIO_CPP_SERVER_BIN", "/opt/audio.cpp/build/linux-cuda-release/bin/audiocpp_server")
ENGINE_URL = "http://127.0.0.1:8081"
ACTIVE_MODEL = os.environ.get("AUDIO_CPP_ACTIVE_MODEL", "qwen3-tts-1.7b-base-bf16")
VOICE_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "/voices"))
MIN_FREE_MIB = {"qwen3-tts-0.6b-base-bf16": 5500, "qwen3-tts-1.7b-base-bf16": 10500}
MAX_BUSY_PERCENT = 85
MIN_SYNTHESIS_FREE_MIB = 2048
MAX_SYNTHESIS_BUSY_PERCENT = 95
GPU_GUARD_SETTINGS_PATH = VOICE_LIBRARY_DIR / "candidate_gpu_guard.json"
GPU_GUARD_MODES = {"enforced", "custom", "disabled"}
EVENTS: deque[dict[str, Any]] = deque(maxlen=20)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("audio_cpp_candidate")
app = FastAPI(title="audio.cpp candidate supervisor")
engine: subprocess.Popen[bytes] | None = None
switch_lock = asyncio.Lock()
generation_lock = asyncio.Lock()
state: dict[str, Any] = {
    "activeModel": ACTIVE_MODEL,
    "state": "loading",
    "reason": None,
    "lastError": None,
    "lastLoadElapsedS": None,
    "gpu": None,
    "lastAction": None,
}


def _default_gpu_guard_settings() -> dict[str, int | str]:
    return {
        "mode": "enforced",
        # 1.7B is the candidate default, so expose its safe reserve as the
        # editable Custom starting point. Enforced mode remains model-specific.
        "load_min_free_mib": MIN_FREE_MIB[ACTIVE_MODEL],
        "synthesis_min_free_mib": MIN_SYNTHESIS_FREE_MIB,
        "load_max_utilization_percent": MAX_BUSY_PERCENT,
        "synthesis_max_utilization_percent": MAX_SYNTHESIS_BUSY_PERCENT,
    }


def _load_gpu_guard_settings() -> dict[str, int | str]:
    settings = _default_gpu_guard_settings()
    try:
        saved = json.loads(GPU_GUARD_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if isinstance(saved, dict):
        settings.update({key: saved[key] for key in settings if key in saved})
    return settings


gpu_guard_settings: dict[str, int | str] = _load_gpu_guard_settings()


def _save_gpu_guard_settings() -> None:
    """Atomically persist candidate-only GPU admission preferences."""
    VOICE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    temporary = GPU_GUARD_SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(gpu_guard_settings, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(GPU_GUARD_SETTINGS_PATH)


def _guard_policy(model_id: str, operation: str) -> tuple[int, int, bool]:
    """Return free-VRAM threshold, utilization ceiling, and manual bypass flag."""
    mode = str(gpu_guard_settings["mode"])
    if operation == "synthesis":
        default_required = MIN_SYNTHESIS_FREE_MIB
        configured_required = int(gpu_guard_settings["synthesis_min_free_mib"])
        configured_utilization = int(gpu_guard_settings["synthesis_max_utilization_percent"])
    else:
        default_required = MIN_FREE_MIB[model_id]
        configured_required = int(gpu_guard_settings["load_min_free_mib"])
        configured_utilization = int(gpu_guard_settings["load_max_utilization_percent"])
    required = configured_required if mode == "custom" else default_required
    return required, configured_utilization, mode == "disabled"


def _record_event(action: str, *, model_id: str | None = None, **details: Any) -> dict[str, Any]:
    """Keep a small user-visible lifecycle trail without exposing request data."""
    event = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": action,
        "model": model_id or state.get("activeModel"),
        **details,
    }
    EVENTS.append(event)
    state["lastAction"] = action
    logger.info("candidate action=%s model=%s details=%s", action, event["model"], details)
    return event


def _template() -> dict[str, Any]:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _models() -> dict[str, dict[str, Any]]:
    return {str(model["id"]): model for model in _template()["models"]}


def gpu_guard(
    model_id: str,
    minimum_free_mib: int | None = None,
    max_busy_percent: int = MAX_BUSY_PERCENT,
    *,
    operation: str = "load",
    apply_policy: bool = True,
) -> dict[str, Any]:
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
        free_mib, utilization = (int(value.strip()) for value in line.split(",")[:2])
    except Exception as exc:
        if apply_policy and str(gpu_guard_settings["mode"]) == "disabled":
            return {
                "ok": True,
                "bypassed": True,
                "reason": f"GPU guard manually disabled; GPU status unavailable ({type(exc).__name__}) and operation will be attempted.",
                "operation": operation,
                "guardMode": "disabled",
            }
        return {"ok": False, "reason": f"GPU status unavailable: {type(exc).__name__}: {exc}"}
    policy_required, policy_utilization, bypassed = _guard_policy(model_id, operation)
    required = minimum_free_mib if minimum_free_mib is not None else policy_required
    utilization_limit = max_busy_percent if minimum_free_mib is not None else policy_utilization
    details = {
        "freeMiB": free_mib,
        "requiredMiB": required,
        "utilizationPercent": utilization,
        "maxUtilizationPercent": utilization_limit,
        "operation": operation,
        "guardMode": gpu_guard_settings["mode"],
    }
    if apply_policy and bypassed:
        return {
            "ok": True,
            "bypassed": True,
            "reason": "GPU guard manually disabled by candidate user; load or synthesis may still fail if VRAM is exhausted.",
            **details,
        }
    if free_mib < required or utilization >= utilization_limit:
        return {"ok": False, "reason": "GPU busy/insufficient VRAM; operation was not attempted.", **details}
    return {"ok": True, **details}


def _reconcile_engine() -> None:
    """Report a child killed by external GPU pressure as an error, not unload."""
    global engine
    if state["state"] == "loaded" and (engine is None or engine.poll() is not None):
        exit_code = engine.returncode if engine is not None else None
        message = f"audio.cpp child exited unexpectedly (exit={exit_code})."
        state.update(state="evicted", reason=message, lastError=message)
        _record_event("child-exit", exitCode=exit_code, gpu=gpu_guard(state["activeModel"], minimum_free_mib=0, apply_policy=False))
        engine = None


def _stop_engine(*, action: str = "release") -> None:
    global engine
    if engine and engine.poll() is None:
        engine.terminate()
        try:
            engine.wait(timeout=20)
        except subprocess.TimeoutExpired:
            engine.kill(); engine.wait(timeout=10)
    engine = None
    _record_event(action)


def _start_engine(model_id: str) -> None:
    global engine
    model = dict(_models()[model_id])
    model["session_options"] = {
        **dict(model.get("session_options") or {}),
        "qwen3_tts.mem_saver": "true",
    }
    config = _template(); config["port"] = 8081; config["models"] = [model]
    ACTIVE_CONFIG.write_text(json.dumps(config), encoding="utf-8")
    engine = subprocess.Popen([ENGINE_BIN, "--config", str(ACTIVE_CONFIG)])
    _record_event("child-start", model_id=model_id, memSaver=True)


async def _engine_ready() -> bool:
    async with httpx.AsyncClient(timeout=1.0) as client:
        try: return (await client.get(f"{ENGINE_URL}/health")).is_success
        except httpx.HTTPError: return False


async def switch_model(model_id: str) -> dict[str, Any]:
    if model_id not in _models(): raise HTTPException(status_code=404, detail="Unknown candidate model.")
    async with switch_lock:
        _reconcile_engine()
        if state["state"] == "loaded" and state["activeModel"] == model_id and await _engine_ready():
            _record_event("load-noop", model_id=model_id, reason="selected model is already resident")
            return {**state, "singleResident": True, "events": list(EVENTS)}
        guard = gpu_guard(model_id, operation="load")
        if not guard["ok"]:
            state.update(reason=guard["reason"], lastError=guard["reason"])
            _record_event("gpu-blocked-load", model_id=model_id, gpu=guard)
            raise HTTPException(status_code=409, detail={"state": "blocked", **guard})
        async with generation_lock:
            state.update(state="loading", reason=None, lastError=None)
            _record_event("load-requested", model_id=model_id, gpu=guard)
            if engine is not None:
                _stop_engine(action="old-model-released")  # release old model before the new child exists
            started = time.monotonic()
            _start_engine(model_id)
            for _ in range(120):
                if await _engine_ready():
                    break
                await asyncio.sleep(0.25)
            else:
                _stop_engine(action="load-failed-release")
                message = "audio.cpp did not become healthy after the model switch."
                state.update(activeModel=model_id, state="error", reason=message, lastError=message)
                raise HTTPException(status_code=503, detail={"state": "error", "message": message})
            state.update(
                activeModel=model_id,
                state="loaded",
                reason=None,
                lastError=None,
                lastLoadElapsedS=round(time.monotonic() - started, 3),
                gpu=guard,
            )
            _record_event("model-ready", model_id=model_id, loadElapsedS=state["lastLoadElapsedS"], gpu=guard)
        return {**state, "singleResident": True, "events": list(EVENTS)}


@app.on_event("startup")
async def startup() -> None:
    # The engine is deliberately not started until Voice Studio explicitly
    # loads a model. This makes the initial resident set truthful and empty.
    state["state"] = "unloaded"
    _record_event("supervisor-started", model_id=ACTIVE_MODEL)


@app.on_event("shutdown")
async def shutdown() -> None: _stop_engine(action="supervisor-shutdown")


@app.get("/health")
async def health() -> dict[str, Any]:
    _reconcile_engine()
    loaded = state["state"] == "loaded" and await _engine_ready()
    return {
        "status": "ok",
        "backend": {
            "name": "audio.cpp CUDA candidate",
            "model_id": state["activeModel"] if loaded else None,
            "current_model_key": state["activeModel"],
            "loaded_models": [state["activeModel"]] if loaded else [],
            "runtime": {
                "state": state["state"],
                "last_error": state["lastError"],
                "last_load_elapsed_s": state["lastLoadElapsedS"],
                "gpu": state["gpu"],
                "events": list(EVENTS),
                "mem_saver": True,
                "native_incremental_pcm": False,
                "progressive_phrase_pcm": True,
                "sample_rate": 24000,
                "sample_format": "pcm_s16le",
                "synthesis_headroom_mib": _guard_policy(str(state["activeModel"]), "synthesis")[0],
                "gpu_guard": dict(gpu_guard_settings),
            },
        },
        **state,
        "singleResident": True,
        "nativeIncrementalPcm": False,
        "ttsMode": "offline-buffered",
    }


@app.get("/control/status")
async def control_status() -> dict[str, Any]:
    _reconcile_engine()
    return {
        **state,
        "singleResident": True,
        "availableModels": list(_models()),
        "events": list(EVENTS),
        "gpu_guard": dict(gpu_guard_settings),
    }


@app.post("/control/switch")
async def control_switch(payload: dict[str, str]) -> dict[str, Any]: return await switch_model(str(payload.get("model") or ""))


def _update_gpu_guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist explicit candidate-only GPU admission settings."""
    mode = str(payload.get("mode", gpu_guard_settings["mode"])).strip().lower()
    if mode not in GPU_GUARD_MODES:
        raise HTTPException(status_code=422, detail="GPU guard mode must be enforced, custom, or disabled.")
    updated = dict(gpu_guard_settings)
    updated["mode"] = mode
    for key in ("load_min_free_mib", "synthesis_min_free_mib"):
        if key in payload:
            try:
                value = int(payload[key])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"{key} must be an integer.") from exc
            if not 0 <= value <= 16384:
                raise HTTPException(status_code=422, detail=f"{key} must be between 0 and 16384 MiB.")
            updated[key] = value
    for key in ("load_max_utilization_percent", "synthesis_max_utilization_percent"):
        if key in payload:
            try:
                value = int(payload[key])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"{key} must be an integer.") from exc
            if not 1 <= value <= 100:
                raise HTTPException(status_code=422, detail=f"{key} must be between 1 and 100 percent.")
            updated[key] = value
    gpu_guard_settings.clear()
    gpu_guard_settings.update(updated)
    _save_gpu_guard_settings()
    _record_event("gpu-guard-updated", mode=mode, settings=dict(gpu_guard_settings))
    return {"gpu_guard": dict(gpu_guard_settings), "warning": "Disabled mode bypasses admission checks only; it cannot prevent a CUDA out-of-memory failure."}


@app.get("/control/gpu-guard")
async def control_gpu_guard() -> dict[str, Any]:
    return {"gpu_guard": dict(gpu_guard_settings)}


@app.post("/control/gpu-guard")
async def control_gpu_guard_update(payload: dict[str, Any]) -> dict[str, Any]:
    return _update_gpu_guard(payload)


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    active = str(state["activeModel"])
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "engine", "family": "qwen3_tts", "task": "tts", "mode": "offline", "active": model_id == active} for model_id in _models()]}


@app.get("/v1/voices")
async def voices() -> dict[str, Any]:
    """Candidate-private Base clone inventory for compatible callers."""
    return {"object": "list", "data": _voice_profiles()}


@app.post("/v1/voices/profiles/{profile_id}")
async def import_voice_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _write_candidate_profile(profile_id, payload)


@app.get("/v1/backend/models")
async def backend_models() -> dict[str, Any]:
    _reconcile_engine()
    active = str(state["activeModel"])
    loaded = state["state"] == "loaded" and await _engine_ready()
    return {
        "available": list(_models()),
        "current": active,
        "loaded_models": [active] if loaded else [],
        "state": "loaded" if loaded else state["state"],
        "last_error": state["lastError"],
        "runtime": {
            "last_load_elapsed_s": state["lastLoadElapsedS"],
            "gpu": state["gpu"],
            "gpu_now": gpu_guard(active, minimum_free_mib=0, apply_policy=False),
            "load_headroom_mib": _guard_policy(active, "load")[0],
            "synthesis_headroom_mib": _guard_policy(active, "synthesis")[0],
            "gpu_guard": dict(gpu_guard_settings),
            "mem_saver": True,
            "native_incremental_pcm": False,
            "progressive_phrase_pcm": True,
            "sample_rate": 24000,
            "sample_format": "pcm_s16le",
            "tts_mode": "offline-buffered",
        },
        "events": list(EVENTS),
        "last_action": state["lastAction"],
        "singleResident": True,
    }


@app.post("/v1/backend/models/switch")
async def backend_switch(payload: dict[str, str]) -> dict[str, Any]:
    return await switch_model(str(payload.get("model_key") or ""))


@app.post("/v1/backend/models/unload")
async def backend_unload() -> dict[str, Any]:
    async with switch_lock:
        async with generation_lock:
            _stop_engine(action="explicit-unload")
            state.update(state="unloaded", reason="Model released by Voice Studio.", lastError=None)
        return {"current": state["activeModel"], "state": "unloaded", "loaded_models": [], "singleResident": True, "events": list(EVENTS)}


def _apply_clone_profile(payload: dict[str, Any]) -> None:
    voice = str(payload.get("voice") or "")
    if not voice.startswith("clone:"):
        return
    profile_id = voice.removeprefix("clone:")
    profile_root = VOICE_LIBRARY_DIR / "profiles" / profile_id
    try:
        meta = json.loads((profile_root / "meta.json").read_text(encoding="utf-8"))
        reference = profile_root / str(meta["ref_audio_filename"])
    except (OSError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"Clone profile `{profile_id}` is unavailable: {exc}") from exc
    if not reference.is_file():
        raise HTTPException(status_code=404, detail=f"Clone profile `{profile_id}` is missing its reference audio.")
    payload.update(
        task_type="Base",
        ref_audio=base64.b64encode(reference.read_bytes()).decode("ascii"),
        ref_text=str(meta.get("ref_text") or ""),
        x_vector_only_mode=bool(meta.get("x_vector_only_mode")),
    )


def _normalize_gradio_seed(payload: dict[str, Any]) -> None:
    """Translate Gradio's negative random-seed sentinel for audio.cpp."""
    seed = payload.get("seed")
    if seed is None:
        return
    try:
        normalized_seed = int(seed)
    except (TypeError, ValueError):
        payload.pop("seed", None)
    else:
        if normalized_seed < 0:
            payload.pop("seed", None)
        else:
            payload["seed"] = normalized_seed


def _container_reachable_llm_endpoint(endpoint: str) -> str:
    """Map a Studio user's host-loopback endpoint into the candidate network."""
    parsed = urlsplit(endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return endpoint
    host = "host.docker.internal"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def _llamacpp_audio_part(data_uri: str) -> dict[str, Any]:
    """Convert browser-recorded data URIs to llama.cpp's input_audio shape."""
    header, separator, encoded = data_uri.partition(",")
    if not separator or not header.startswith("data:audio/") or ";base64" not in header:
        raise HTTPException(status_code=422, detail="Streaming Playground requires base64 audio data.")
    audio_format = header.removeprefix("data:audio/").split(";", 1)[0].lower()
    if audio_format in {"x-wav", "wave"}:
        audio_format = "wav"
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}}


def _llamacpp_history(messages: list[Any]) -> list[dict[str, Any]]:
    """Keep text history valid when earlier Studio turns originated from audio."""
    safe: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role not in {"user", "assistant", "system"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            safe.append({"role": role, "content": content})
        elif role == "user" and message.get("audio_data_url"):
            safe.append({"role": "user", "content": "[Earlier spoken user turn]"})
    return safe


def _voice_profiles() -> list[dict[str, Any]]:
    """Expose only candidate-private Base profiles through the OpenAI boundary."""
    profiles: list[dict[str, Any]] = []
    root = VOICE_LIBRARY_DIR / "profiles"
    if not root.is_dir():
        return profiles
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            profile_id = meta_path.parent.name
            reference = meta_path.parent / str(meta.get("ref_audio_filename") or "ref_audio.wav")
            if reference.is_file():
                profiles.append({
                    "id": profile_id,
                    "voice": f"clone:{profile_id}",
                    "name": str(meta.get("name") or profile_id),
                    "task": str(meta.get("task_type") or "Base"),
                    "language": str(meta.get("language") or "Auto"),
                })
        except (OSError, ValueError):
            continue
    return profiles


def _write_candidate_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically import a compatible Base profile into the private candidate library."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", profile_id):
        raise HTTPException(status_code=422, detail="Invalid clone profile id.")
    encoded = str(payload.get("ref_audio") or "")
    if not encoded:
        raise HTTPException(status_code=422, detail="Clone reference audio is required.")
    try:
        audio = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid clone reference audio.") from exc
    if not audio:
        raise HTTPException(status_code=422, detail="Clone reference audio is empty.")
    profile_dir = VOICE_LIBRARY_DIR / "profiles" / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    reference_name = "ref_audio.wav"
    temp_audio = profile_dir / f".{reference_name}.tmp"
    temp_meta = profile_dir / ".meta.json.tmp"
    temp_audio.write_bytes(audio)
    temp_audio.replace(profile_dir / reference_name)
    metadata = {
        "profile_id": profile_id,
        "name": str(payload.get("name") or profile_id),
        "task_type": "Base",
        "language": str(payload.get("language") or "Auto"),
        "ref_text": str(payload.get("ref_text") or ""),
        "x_vector_only_mode": bool(payload.get("x_vector_only_mode")),
        "ref_audio_filename": reference_name,
    }
    temp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temp_meta.replace(profile_dir / "meta.json")
    _record_event("profile-imported", profile_id=profile_id)
    return {"id": profile_id, "voice": f"clone:{profile_id}", "name": metadata["name"]}


def _render_master_wav(master_wav: bytes, requested_format: str) -> tuple[bytes, str, str, dict[str, int | str]]:
    """Return exactly one requested output, using audio.cpp's complete WAV as master."""
    fmt = (requested_format or "wav").lower()
    try:
        with wave.open(io.BytesIO(master_wav), "rb") as reader:
            metadata: dict[str, int | str] = {
                "codec": "pcm_s16le",
                "container": "WAV",
                "quality": "native lossless master",
                "sampleRate": reader.getframerate(),
                "channels": reader.getnchannels(),
                "bitsPerSample": reader.getsampwidth() * 8,
                "durationSeconds": round(reader.getnframes() / reader.getframerate(), 3),
            }
            if fmt == "pcm":
                metadata["container"] = "raw PCM"
                metadata["quality"] = "native PCM extracted without re-encoding"
                return reader.readframes(reader.getnframes()), "audio/pcm", "pcm", metadata
    except wave.Error as exc:
        raise HTTPException(status_code=502, detail=f"audio.cpp returned an invalid WAV master: {exc}") from exc
    if fmt == "wav":
        return master_wav, "audio/wav", "wav", metadata

    encoders = {
        # Never resample or downmix the native WAV master.  Lossless formats
        # preserve it bit-for-bit; lossy formats use intentionally high rates.
        "mp3": ("libmp3lame", ["-b:a", "320k"], "audio/mpeg", "mp3", "high-quality 320 kbps"),
        "flac": ("flac", ["-compression_level", "8"], "audio/flac", "flac", "lossless"),
        "aac": ("aac", ["-b:a", "320k"], "audio/aac", "aac", "high-quality 320 kbps"),
        "opus": ("libopus", ["-b:a", "320k"], "audio/ogg", "opus", "high-quality 320 kbps"),
    }
    if fmt not in encoders:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format `{fmt}`.")
    encoder, encoder_args, media_type, extension, quality = encoders[fmt]
    with tempfile.TemporaryDirectory(prefix="audio-cpp-output-") as directory:
        source = Path(directory) / "master.wav"
        output = Path(directory) / f"output.{extension}"
        source.write_bytes(master_wav)
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:a", encoder, *encoder_args, str(output)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not output.exists():
            raise HTTPException(status_code=502, detail=f"Failed to encode `{fmt}` output: {result.stderr.strip()}")
        metadata["codec"] = encoder
        metadata["container"] = extension.upper()
        metadata["quality"] = quality
        return output.read_bytes(), media_type, extension, metadata


@app.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    payload = await request.json()
    _apply_clone_profile(payload)
    async with generation_lock:
        model_id = str(payload.get("model") or state["activeModel"])
        if model_id != state["activeModel"]: raise HTTPException(status_code=409, detail="Selected model is not active; switch it in Voice Studio before synthesis.")
        if state["state"] != "loaded":
            raise HTTPException(status_code=409, detail="No candidate model is loaded. Use Voice Studio model controls first.")
        guard = gpu_guard(
            model_id,
            operation="synthesis",
        )
        if not guard["ok"]:
            state.update(reason=guard["reason"], lastError=guard["reason"])
            _record_event("gpu-blocked-synthesis", model_id=model_id, gpu=guard)
            raise HTTPException(status_code=409, detail={"state": "blocked", **guard})
        if not await _engine_ready(): raise HTTPException(status_code=503, detail="Candidate model is switching/loading; retry when Voice Studio reports loaded.")
        if payload.get("task_type") == "Base" or payload.get("ref_audio"):
            _record_event("generation-start", model_id=model_id, requestedFormat=str(payload.get("response_format") or "wav"))
            return await _voice_clone_response(payload)
        payload["stream"] = False
        async with httpx.AsyncClient(timeout=600.0) as client: response = await client.post(f"{ENGINE_URL}/v1/audio/speech", json=payload)
        return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type"))


@app.post("/v1/audio/voice-clone")
async def voice_clone(request: Request) -> Response:
    """Compatibility adapter for the copied Gradio Base-profile playground."""
    payload = await request.json(); model_id = str(payload.get("model") or state["activeModel"])
    if model_id != state["activeModel"]:
        raise HTTPException(status_code=409, detail="Load the selected model before generation.")
    if state["state"] != "loaded":
        raise HTTPException(status_code=409, detail="No candidate model is loaded. Use Voice Studio model controls first.")
    guard = gpu_guard(
        model_id,
        operation="synthesis",
    )
    if not guard["ok"]:
        state.update(reason=guard["reason"], lastError=guard["reason"])
        _record_event("gpu-blocked-synthesis", model_id=model_id, gpu=guard)
        raise HTTPException(status_code=409, detail={"state": "blocked", **guard})
    async with generation_lock:
        _record_event("generation-start", model_id=model_id, requestedFormat=str(payload.get("response_format") or "wav"))
        return await _voice_clone_response(payload)


async def _voice_clone_response(payload: dict[str, Any]) -> Response:
    raw = str(payload.get("ref_audio") or "")
    if not raw:
        raise HTTPException(status_code=400, detail="A Base clone reference WAV is required.")
    try:
        reference = Path("/tmp/gradio-clone-reference.wav")
        reference.write_bytes(base64.b64decode(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid clone reference audio: {exc}") from exc
    engine_payload = dict(payload)
    # Gradio uses -1 as its "random seed" sentinel. audio.cpp validates seed
    # as unsigned, so omission is the compatible way to request randomness.
    _normalize_gradio_seed(engine_payload)
    requested_format = str(payload.get("response_format") or "wav").lower()
    engine_payload.update(
        model=state["activeModel"],
        voice_ref=str(reference),
        reference_text=str(payload.get("ref_text") or ""),
        response_format="wav",
        stream=False,
    )
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(f"{ENGINE_URL}/v1/audio/speech", json=engine_payload)
    if response.is_error:
        _record_event("generation-error", model_id=state["activeModel"], status=response.status_code)
        return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type"))
    content, media_type, extension, metadata = _render_master_wav(response.content, requested_format)
    _record_event(
        "generation-complete",
        model_id=state["activeModel"],
        requestedFormat=requested_format,
        returnedFormat=extension,
        bytes=len(content),
        durationSeconds=metadata.get("durationSeconds"),
    )
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "X-TTS-Model": state["activeModel"],
            "X-TTS-Codec": str(metadata["codec"]),
            "X-TTS-Container": str(metadata["container"]),
            "X-TTS-Quality": str(metadata["quality"]),
            "X-TTS-Sample-Rate": str(metadata["sampleRate"]),
            "X-TTS-Bits-Per-Sample": str(metadata["bitsPerSample"]),
            "X-TTS-Duration-Seconds": str(metadata["durationSeconds"]),
            "X-TTS-Format": extension,
        },
    )


@app.post("/v1/voice-studio/llamacpp-audio-turn/stream")
async def llamacpp_audio_turn(request: Request) -> StreamingResponse:
    """Pass through the Studio's user-configured llama.cpp SSE turn."""
    payload = await request.json()
    endpoint = _container_reachable_llm_endpoint(str(payload.pop("endpoint", "")).strip())
    api_key = str(payload.pop("api_key", "")).strip()
    if not endpoint.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="A valid llama.cpp HTTP endpoint is required.")
    messages = [{"role": "system", "content": payload.pop("system_prompt", "")}]
    messages.extend(_llamacpp_history(payload.pop("history", [])))
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": payload.pop("prompt", "Respond to the spoken message.")},
                _llamacpp_audio_part(str(payload.pop("audio_data_url", ""))),
            ],
        }
    )
    upstream = {"model": payload.pop("model", ""), "messages": messages, "stream": True, **payload}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", endpoint, json=upstream, headers=headers) as response:
                if response.is_error:
                    body = await response.aread()
                    yield f'data: {{"error": {{"message": {json.dumps(body.decode(errors="replace"))}}}}}\n\n'.encode()
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
