"""
Gradio Voice Studio for Qwen3-TTS.

This module defines a comprehensive Gradio-based UI that allows users to
create, manage, and export reusable voice profiles using the Qwen3-TTS
API.  The interface supports three primary workflows: preset voices
(CustomVoice), voice design (VoiceDesign), and voice cloning (Base).
Users can store profiles locally, preview and delete them, export
selected profiles to ZIP archives, and synthesize speech with saved
profiles via an interactive playground.

The entrypoint function ``build_app(base_url: str, library_dir: Path)``
returns a Gradio Blocks object configured with the desired API base URL
and profile storage directory.  See README or docstrings for usage.
"""

import argparse
import base64
import json
import mimetypes
import os
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx


# -----------------------------------------------------------------------------
# Configuration and Defaults
# -----------------------------------------------------------------------------

# The default directory used to store voice profiles.  This can be overridden
# via the VOICE_LIBRARY_DIR environment variable or passed directly to
# ``build_app``.
DEFAULT_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")).resolve()

# Base URL pointing to the running Qwen3-TTS API.  If not provided, this
# defaults to the local server on port 8880.  It may be overridden via the
# TTS_BASE_URL environment variable or passed to ``build_app``.
DEFAULT_TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8880").rstrip("/")

# Default timeout for API requests in seconds.  Can be customized via the
# TTS_TIMEOUT_S environment variable or overridden at runtime.
DEFAULT_TIMEOUT_S = float(os.environ.get("TTS_TIMEOUT_S", "300"))

# Supported task types defined by the Qwen3-TTS API.  These values map to
# endpoint parameters used when creating and managing profiles.
SUPPORTED_TASK_TYPES = ["CustomVoice", "VoiceDesign", "Base"]

# A sample reference line used for voice design workflows.  This line will be
# synthesized as part of the profile creation process.
DEFAULT_REFERENCE_LINE = (
    "Hi! This is a reference clip for my custom voice. "
    "I will reuse this voice for future speech synthesis."
)

# Fallback voices list used when the server does not provide a voices
# endpoint or fails to return names.
FALLBACK_VOICES = ["Vivian", "Ryan", "Serena", "Dylan", "Eric", "Aiden"]


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    """Retained as a no-op compatibility hook for the recovered Studio."""


@dataclass
class VoiceProfile:
    """Representation of a saved voice profile.

    Attributes:
        profile_id: Unique identifier for the profile.
        name: Human-friendly name assigned by the user.
        task_type: Type of profile (CustomVoice, VoiceDesign, Base).
        created_at: ISO8601 timestamp indicating when the profile was created.
        language: Language code or Auto for automatic detection.
        voice: Name of the voice (for CustomVoice or design clones).
        instructions: Style instructions (for CustomVoice or voice design).
        ref_text: Transcript of the reference audio (for Base profiles).
        x_vector_only_mode: Whether the clone is created without a transcript.
        ref_audio_filename: Filename of the reference audio stored in the profile directory.
        origin: Human-readable descriptor of how the profile was created.
    """

    profile_id: str
    name: str
    task_type: str
    created_at: str
    language: str = "Auto"
    voice: str = "Vivian"
    instructions: str = ""
    ref_text: str = ""
    x_vector_only_mode: bool = False
    ref_audio_filename: str = ""
    origin: str = ""


def ensure_dirs(library_dir: Path) -> Dict[str, Path]:
    """Ensure that the profiles and exports directories exist.

    Returns a dictionary mapping directory names to Path objects.
    """
    profiles_dir = library_dir / "profiles"
    exports_dir = library_dir / "exports"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    return {"profiles": profiles_dir, "exports": exports_dir}


def now_iso() -> str:
    """Return the current UTC time as an ISO8601-formatted string."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_profile_id() -> str:
    """Generate a short, unique profile identifier."""
    return uuid.uuid4().hex[:12]


def profile_dir(library_dir: Path, profile_id: str) -> Path:
    """Return the path to a given profile's directory."""
    return ensure_dirs(library_dir)["profiles"] / profile_id


def meta_path(library_dir: Path, profile_id: str) -> Path:
    """Return the path to a profile's metadata file."""
    return profile_dir(library_dir, profile_id) / "meta.json"


def load_profile(library_dir: Path, profile_id: str) -> VoiceProfile:
    """Load a profile from disk into a VoiceProfile instance."""
    p = meta_path(library_dir, profile_id)
    data = json.loads(p.read_text(encoding="utf-8"))
    return VoiceProfile(**data)


def save_profile(library_dir: Path, vp: VoiceProfile) -> None:
    """Persist profile metadata with an atomic same-directory replacement."""
    d = profile_dir(library_dir, vp.profile_id)
    d.mkdir(parents=True, exist_ok=True)
    target = d / "meta.json"
    temporary = d / ".meta.json.tmp"
    temporary.write_text(json.dumps(vp.__dict__, indent=2), encoding="utf-8")
    temporary.replace(target)


def delete_profile(library_dir: Path, profile_id: str) -> None:
    """Remove a profile directory and all its contents."""
    d = profile_dir(library_dir, profile_id)
    if d.exists():
        shutil.rmtree(d)


def list_profiles(library_dir: Path) -> List[VoiceProfile]:
    """Return all profiles stored in the library sorted by creation time."""
    dirs = ensure_dirs(library_dir)["profiles"]
    out: List[VoiceProfile] = []
    for child in sorted(dirs.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            mp = child / "meta.json"
            if mp.exists():
                try:
                    data = json.loads(mp.read_text(encoding="utf-8"))
                    out.append(VoiceProfile(**data))
                except Exception:
                    pass  # skip corrupted entries
    # Sort newest first
    out.sort(key=lambda x: x.created_at, reverse=True)
    return out


def profiles_table_rows(profiles: List[VoiceProfile]) -> List[List[Any]]:
    """Convert profiles list into table rows for the library tab."""
    rows = []
    for p in profiles:
        rows.append([
            p.profile_id,
            p.name,
            p.task_type,
            p.origin,
            p.language,
            p.voice,
            (p.instructions[:60] + "…") if len(p.instructions) > 60 else p.instructions,
            "yes" if bool(p.ref_audio_filename) else "no",
            p.created_at,
        ])
    return rows


def normalize_base_url(base_url: str) -> str:
    """Ensure the base URL does not end with a trailing slash."""
    return base_url.rstrip("/")


def data_uri_from_file(file_path: Path) -> str:
    """Encode a file's contents as a data URI with guessed MIME type."""
    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime:
        mime = "audio/wav"
    b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def base64_from_file(file_path: Path) -> str:
    """Encode a file's contents as base64 string (no data URI prefix)."""
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def write_bytes_to_temp_audio(content: bytes, ext: str) -> str:
    """Write raw audio bytes to a temporary file and return its path."""
    ext = ext.lstrip(".")
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    os.close(fd)
    Path(path).write_bytes(content)
    return path


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Convert raw PCM bytes to WAV format with proper header."""
    import struct
    import io
    
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = len(pcm_bytes)
    
    buffer = io.BytesIO()
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size - 8
    buffer.write(b'WAVE')
    
    # Format chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # Data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(pcm_bytes)
    
    return buffer.getvalue()


def request_tts(base_url: str, payload: Dict[str, Any], timeout_s: float) -> Tuple[bytes, str]:
    """Call the /v1/audio/speech endpoint and return audio bytes and extension."""
    url = normalize_base_url(base_url) + "/v1/audio/speech"
    response_format = payload.get("response_format") or "wav"
    payload["response_format"] = response_format
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
    ext = (r.headers.get("x-tts-format") or response_format).lower()
    if ext == "pcm":
        ext = "raw"
    return r.content, ext


def request_tts_voice_clone(
    base_url: str, payload: Dict[str, Any], timeout_s: float
) -> Tuple[bytes, str, Dict[str, Any]]:
    """Call the /v1/audio/voice-clone endpoint (non-streaming). Returns (audio_bytes, extension, headers_dict)."""
    url = normalize_base_url(base_url) + "/v1/audio/voice-clone"
    response_format = payload.get("response_format") or "wav"
    payload["response_format"] = response_format
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
    ext = (r.headers.get("x-tts-format") or response_format).lower()
    if ext == "pcm":
        ext = "raw"
    headers = dict(r.headers) if r.headers else {}
    return r.content, ext, headers


def request_tts_streaming(
    base_url: str, 
    payload: Dict[str, Any], 
    timeout_s: float
) -> Tuple[bytes, str, Dict[str, Any]]:
    """
    Call the /v1/audio/voice-clone/stream endpoint and return audio bytes, extension, and timing info.
    
    Returns:
        Tuple of (audio_bytes, extension, timing_info)
    """
    raise RuntimeError(
        "audio.cpp Qwen3-TTS has no native incremental PCM endpoint; use buffered streaming scaffolding instead."
    )
    import struct
    import json as json_module
    
    url = normalize_base_url(base_url) + "/v1/audio/voice-clone/stream"
    
    with httpx.Client(timeout=timeout_s) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            chunks = []
            timing_info = {}
            buffer = b""
            
            for chunk in response.iter_bytes():
                buffer += chunk
                
                # Process complete chunks
                # Format: [4 bytes JSON length][JSON metadata][4 bytes audio length][audio bytes]
                while len(buffer) >= 4:
                    json_len = struct.unpack('<I', buffer[:4])[0]
                    if len(buffer) < 4 + json_len + 4:
                        break  # Need more data for JSON + audio length
                    
                    json_bytes = buffer[4:4 + json_len]
                    metadata = json_module.loads(json_bytes.decode('utf-8'))
                    audio_len = struct.unpack('<I', buffer[4 + json_len:4 + json_len + 4])[0]
                    
                    if len(buffer) < 4 + json_len + 4 + audio_len:
                        break  # Need more data for audio bytes
                    
                    audio_bytes = buffer[4 + json_len + 4:4 + json_len + 4 + audio_len]
                    buffer = buffer[4 + json_len + 4 + audio_len:]  # Keep remaining data
                    
                    if metadata.get("error"):
                        raise RuntimeError(metadata["error"])
                    
                    if metadata.get("done"):
                        timing_info = {
                            "first_chunk_time": metadata.get("first_chunk_time"),
                            "total_time": metadata.get("total_time"),
                            "audio_duration": metadata.get("audio_duration"),
                            "rtf": metadata.get("rtf"),
                            "chunk_count": metadata.get("chunk_count"),
                            "seed_used": metadata.get("seed_used"),
                        }
                    elif len(audio_bytes) > 0:
                        chunks.append(audio_bytes)
            
            # Combine all audio chunks (raw PCM data)
            total_length = sum(len(c) for c in chunks)
            combined_pcm = b"".join(chunks)
            
            # Convert raw PCM to WAV format with proper header
            # PCM is 16-bit signed integers at 24000 Hz
            wav_bytes = pcm_to_wav(combined_pcm, sample_rate=24000)
            
            return wav_bytes, "wav", timing_info


def try_fetch_voices(base_url: str, timeout_s: float) -> List[str]:
    """Attempt to fetch available voices from the /v1/voices endpoint."""
    url = normalize_base_url(base_url) + "/v1/voices"
    try:
        with httpx.Client(timeout=min(timeout_s, 20.0)) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
        if isinstance(data, dict) and isinstance(data.get("voices"), list):
            return [str(x) for x in data["voices"]]
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            names: List[str] = []
            for item in data["data"]:
                if isinstance(item, dict) and "name" in item:
                    names.append(str(item["name"]))
            if names:
                return names
    except Exception:
        pass
    return FALLBACK_VOICES


def export_profiles_zip(library_dir: Path, profile_ids: Optional[List[str]] = None) -> str:
    """Create a ZIP archive of selected profiles (or all) and return its path."""
    ensure_dirs(library_dir)
    profiles = list_profiles(library_dir)
    if profile_ids:
        profiles = [p for p in profiles if p.profile_id in set(profile_ids)]
    exports_dir = ensure_dirs(library_dir)["exports"]
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zip_path = exports_dir / f"voices_export_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        manifest: List[Dict[str, Any]] = []
        for p in profiles:
            d = profile_dir(library_dir, p.profile_id)
            z.write(d / "meta.json", arcname=f"{p.profile_id}/meta.json")
            if p.ref_audio_filename:
                ref_path = d / p.ref_audio_filename
                if ref_path.exists():
                    z.write(ref_path, arcname=f"{p.profile_id}/{p.ref_audio_filename}")
            manifest.append(p.__dict__)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    return str(zip_path)


def import_profiles_zip(library_dir: Path, zip_path: Path) -> Dict[str, Any]:
    """Import profiles from an export ZIP and return a summary."""
    if not zip_path.exists():
        raise ValueError(f"ZIP not found: {zip_path}")

    ensure_dirs(library_dir)
    imported = 0
    skipped = 0
    errors: List[str] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        profile_roots = sorted({n.split("/", 1)[0] for n in names if "/" in n and n != "MANIFEST.json"})
        for root in profile_roots:
            meta_name = f"{root}/meta.json"
            if meta_name not in names:
                skipped += 1
                errors.append(f"{root}: missing meta.json")
                continue
            try:
                meta_raw = z.read(meta_name).decode("utf-8")
                meta = json.loads(meta_raw)
            except Exception as exc:
                skipped += 1
                errors.append(f"{root}: invalid meta.json ({exc})")
                continue

            try:
                vp = VoiceProfile(**meta)
            except Exception as exc:
                skipped += 1
                errors.append(f"{root}: invalid profile fields ({exc})")
                continue

            # Always allocate a fresh profile_id to avoid collisions/overwrites.
            new_id = safe_profile_id()
            while profile_dir(library_dir, new_id).exists():
                new_id = safe_profile_id()
            old_ref_filename = vp.ref_audio_filename
            vp.profile_id = new_id

            dest_dir = profile_dir(library_dir, vp.profile_id)
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy referenced audio if present in archive.
            if old_ref_filename:
                ref_member = f"{root}/{old_ref_filename}"
                if ref_member in names:
                    (dest_dir / old_ref_filename).write_bytes(z.read(ref_member))
                else:
                    vp.ref_audio_filename = ""
                    errors.append(f"{root}: referenced audio '{old_ref_filename}' missing in ZIP")

            save_profile(library_dir, vp)
            imported += 1

    return {"imported": imported, "skipped": skipped, "errors": errors}


# -----------------------------------------------------------------------------
# Gradio UI Construction
# -----------------------------------------------------------------------------

# The candidate keeps the copied Faster-side Studio's orange surface.  The
# Gradio theme and the embedded live widgets must use the same palette; CSS
# alone cannot override all native Gradio controls.
def orange_theme() -> Any:
    """Build the copied Studio's orange Gradio theme only when the UI starts."""
    return gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="orange",
        neutral_hue="slate",
    ).set(
        body_background_fill="#fff7ed",
        body_background_fill_dark="#1c1917",
        color_accent="#f97316",
        color_accent_soft="#ffedd5",
        block_title_background_fill="#fff7ed",
        block_title_border_color="#fdba74",
        block_title_text_color="#9a3412",
        button_primary_background_fill="#f97316",
        button_primary_background_fill_hover="#ea580c",
        button_primary_border_color="#f97316",
        button_primary_border_color_hover="#ea580c",
        button_secondary_border_color="#fdba74",
        button_secondary_text_color="#9a3412",
        input_border_color_focus="#f97316",
        checkbox_border_color_focus="#f97316",
        slider_color="#f97316",
    )

# Custom CSS completes the orange theme for the copied Studio-specific layout.
CSS = """
:root {
  --bg0: #fff7ed;
  --bg1: #fffaf5;
  --card: rgba(255,255,255,0.95);
  --card2: rgba(255,255,255,0.98);
  --border: rgba(15,23,42,0.12);
  --text: #0f172a;
  --muted: rgba(15,23,42,0.72);
}
body, .gradio-container { background: radial-gradient(1200px 800px at 10% 10%, var(--bg1), var(--bg0)) !important; }
body, .gradio-container { color: var(--text) !important; color-scheme: light; }
#header {
  padding: 18px 18px;
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(249,115,22,0.16), rgba(251,146,60,0.07));
  border-radius: 18px;
}
.card {
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: 18px;
}
.small { color: var(--muted) !important; font-size: 0.95em; }

/* Keep menus readable without forcing a dark theme */
.gradio-container label, .gradio-container .label, .gradio-container .wrap label { color: var(--muted) !important; }
.gradio-container .prose, .gradio-container .markdown, .gradio-container .md { color: var(--text) !important; }

/* Subtle surfaces: do not override Gradio component colors heavily */
.gradio-container .block, .gradio-container .panel, .gradio-container .gr-box, .gradio-container .wrap {
  border-color: var(--border) !important;
}

/* Streaming widget styles */
.svwidget { font-family: system-ui, sans-serif; }
.svwidget details { margin-bottom: 12px; }
.svwidget summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
.svwidget .sv-settings-grid { display: grid; gap: 8px; padding: 8px 0; }
.svwidget .sv-settings-grid label { display: flex; flex-direction: column; font-size: 0.9em; }
.svwidget .sv-settings-grid input,
.svwidget .sv-settings-grid textarea { padding: 6px 8px; border: 1px solid #d0d0d0; border-radius: 6px; font-size: 0.9em; width: 100%; box-sizing: border-box; }
.svwidget .sv-status { padding: 8px 0; font-size: 0.9em; color: #666; min-height: 1.4em; }
.svwidget .sv-mic-btn { display: inline-flex; align-items: center; gap: 8px; padding: 12px 28px; border-radius: 50px; border: 2px solid #f97316; background: white; color: #c2410c; font-size: 1.05em; cursor: pointer; transition: all 0.2s; }
.svwidget .sv-mic-btn:hover { background: #fff7ed; }
.svwidget .sv-mic-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.svwidget .sv-mic-btn.recording { background: #ef4444; color: white; border-color: #ef4444; animation: sv-pulse 1s infinite; }
@keyframes sv-pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); } 50% { box-shadow: 0 0 0 12px rgba(239,68,68,0); } }
.svwidget .sv-transcript { margin-top: 12px; max-height: 400px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; background: #fafafa; }
.svwidget .sv-msg { margin-bottom: 6px; padding: 6px 10px; border-radius: 8px; }
.svwidget .sv-msg.user { background: #ffedd5; }
.svwidget .sv-msg.assistant { background: #f0fdf4; }
.svwidget .sv-msg .role { font-weight: 600; font-size: 0.8em; color: #666; }
.svwidget .sv-msg .content { margin-top: 2px; word-break: break-word; }
.svwidget .sv-timing { font-size: 0.78em; color: #999; margin-top: 2px; text-align: right; }
.svwidget .sv-clr-btn { margin-top: 8px; padding: 6px 16px; border: 1px solid #d0d0d0; border-radius: 6px; background: white; cursor: pointer; font-size: 0.9em; }
.svwidget .sv-clr-btn:hover { background: #f9f9f9; }
/* Card layout */
.svwidget .sv-card { border: 1px solid var(--border); background: var(--card); border-radius: 18px; margin-bottom: 14px; overflow: hidden; }
.svwidget .sv-card-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 0.95em; }
.svwidget .sv-card-title { display: flex; align-items: center; gap: 6px; }
.svwidget .sv-card-body { padding: 14px 16px; }
.svwidget .sv-card-body-nopad { padding: 0; }
.svwidget .sv-card-badge { font-size: 0.75em; padding: 2px 8px; border-radius: 10px; background: #f0f0f0; color: #666; font-weight: 500; }
.svwidget .sv-card-badge.rec { background: #ef4444; color: white; }
.svwidget .sv-card-badge.speaking { background: #f97316; color: white; }
.svwidget .sv-card-status { font-size: 0.85em; color: #888; margin-top: 6px; min-height: 1.2em; }
.svwidget .sv-mic-select { width: 100%; padding: 8px 10px; border: 1px solid #d0d0d0; border-radius: 8px; font-size: 0.9em; background: white; margin-bottom: 10px; box-sizing: border-box; }
.svwidget .sv-mic-row { display: flex; align-items: center; gap: 10px; }
.svwidget .sv-metrics { display: flex; gap: 16px; font-size: 0.85em; color: #999; margin-top: 8px; }
.svwidget .sv-metrics b { color: #666; }
.svwidget .sv-card-header .sv-clr-btn { margin-top: 0; padding: 4px 12px; font-size: 0.85em; }
"""

TABLE_HEADERS = [
    "id",
    "name",
    "task_type",
    "origin",
    "language",
    "voice",
    "instructions",
    "has_ref_audio",
    "created_at",
]


def build_app(initial_base_url: str, initial_library_dir: Path) -> gr.Blocks:
    """Construct the Gradio Blocks interface for the Voice Studio.

    Args:
        initial_base_url: Base URL of the Qwen3-TTS API to use for requests.
        initial_library_dir: Directory where profiles will be stored.

    Returns:
        A gradio.Blocks object ready to be mounted on a FastAPI server or
        launched standalone via ``.launch()``.
    """
    ensure_dirs(initial_library_dir)
    # Gradio validates event inputs against the component's server-side
    # ``choices`` before it invokes a callback.  A load-event refresh alone is
    # therefore insufficient: the browser can show profile IDs while the
    # server still considers the dropdown empty.  Seed both Playground
    # dropdowns from the private persistent library at construction time.
    initial_profiles = list_profiles(initial_library_dir)
    initial_profile_ids = [profile.profile_id for profile in initial_profiles]
    initial_profile_value = initial_profile_ids[0] if initial_profile_ids else None
    initial_profile_rows = profiles_table_rows(initial_profiles)
    with gr.Blocks(title="Qwen3 Voice Studio", css=CSS, theme=orange_theme()) as demo:
        # Shared state variables
        state_base_url = gr.State(initial_base_url)
        state_library_dir = gr.State(str(initial_library_dir))
        state_voices = gr.State([])

        # Header section
        gr.HTML(
            """
            <div id="header">
              <div style="font-size: 1.35rem; font-weight: 700;">Qwen3 Voice Studio</div>
              <div class="small">
                Create & save reusable voice profiles (preset, designed, or cloned) and export them for inference.
              </div>
            </div>
            """
        )

        # Settings accordion
        with gr.Accordion("Settings & candidate model controls", open=True):
            with gr.Row():
                base_url_in = gr.Textbox(
                    label="TTS Server Base URL",
                    value=initial_base_url,
                    placeholder="http://localhost:8880",
                )
                library_dir_in = gr.Textbox(
                    label="Voice Library Dir",
                    value=str(initial_library_dir),
                    placeholder="./voice_library",
                )
                timeout_in = gr.Number(
                    label="Request timeout (seconds)",
                    value=DEFAULT_TIMEOUT_S,
                    precision=0,
                )
            with gr.Row():
                refresh_voices_btn = gr.Button("Refresh voices from server", variant="primary")
                voices_status = gr.Markdown("", elem_classes=["small"])
            with gr.Row():
                backend_status_md = gr.Markdown("", elem_classes=["small"])
            # Backend model (optimized backend only): show when GET /v1/backend/models returns list
            backend_model_column = gr.Column(visible=False)
            with backend_model_column:
                gr.Markdown("**Backend model controls**")
                backend_model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=[],
                    value=None,
                    # The candidate supervisor remains the authority for
                    # validation.  This avoids Gradio rejecting a live model
                    # choice during its initial asynchronous inventory fetch.
                    allow_custom_value=True,
                )
                with gr.Row():
                    switch_model_btn = gr.Button("Load selected model", variant="secondary")
                    unload_model_btn = gr.Button("Unload model", variant="stop")
                    refresh_model_status_btn = gr.Button("Refresh model status", variant="secondary")
                backend_models_status_md = gr.Markdown("", elem_classes=["small"])
                with gr.Accordion("GPU admission guard (candidate only)", open=False):
                    gr.Markdown(
                        "Use **Custom** to lower the free-VRAM reserve for coexistence tests. "
                        "**Disabled** bypasses this candidate's admission check only; CUDA can still return out-of-memory."
                    )
                    gpu_guard_mode = gr.Radio(
                        ["enforced", "custom", "disabled"],
                        label="Guard mode",
                        value="enforced",
                        info="Enforced uses model-safe defaults. Custom uses the values below.",
                    )
                    with gr.Row():
                        gpu_load_headroom = gr.Number(label="Custom load free-VRAM reserve (MiB)", value=10500, precision=0, minimum=0, maximum=16384)
                        gpu_synthesis_headroom = gr.Number(label="Custom synthesis free-VRAM reserve (MiB)", value=2048, precision=0, minimum=0, maximum=16384)
                    with gr.Row():
                        gpu_load_utilization = gr.Number(label="Load max GPU utilization (%)", value=85, precision=0, minimum=1, maximum=100)
                        gpu_synthesis_utilization = gr.Number(label="Synthesis max GPU utilization (%)", value=95, precision=0, minimum=1, maximum=100)
                    gpu_guard_save_btn = gr.Button("Apply GPU guard settings", variant="secondary")
                    gpu_guard_status_md = gr.Markdown("", elem_classes=["small"])

        # Global log output
        with gr.Row():
            global_log = gr.Markdown("", elem_classes=["small"])

        # Tabs for create, library, and playground
        with gr.Tabs():
            # Create tab and nested subtabs
            with gr.Tab("Create"):
                with gr.Tabs():
                    # Preset (CustomVoice)
                    with gr.Tab("Preset Voice (CustomVoice — unavailable in audio.cpp)"):
                        gr.Markdown("⚠️ CustomVoice checkpoints are not included in this Base-only audio.cpp candidate.")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=320):
                                preset_name = gr.Textbox(label="Profile name", placeholder="e.g. 'Vivian - Friendly NZ Support'")
                                preset_voice = gr.Dropdown(label="Voice", choices=FALLBACK_VOICES, value=FALLBACK_VOICES[0])
                                preset_language = gr.Dropdown(
                                    label="Language",
                                    choices=[
                                        "Auto",
                                        "English",
                                        "Spanish",
                                        "Chinese",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Italian",
                                    ],
                                    value="Auto",
                                )
                                preset_instructions = gr.Textbox(
                                    label="Style instructions (optional)",
                                    placeholder="e.g. Calm, confident, slightly upbeat, clear articulation.",
                                    lines=3,
                                )
                                preset_test_text = gr.Textbox(
                                    label="Test text",
                                    value="Hello! This is my saved preset voice profile.",
                                    lines=3,
                                )
                                preset_generate_btn = gr.Button("Generate (unsupported)", variant="primary", interactive=False)
                                preset_save_btn = gr.Button("Save profile (unsupported)", variant="secondary", interactive=False)
                            with gr.Column(scale=1, min_width=320):
                                preset_audio = gr.Audio(label="Output audio (trimmable)", type="filepath", editable=True)
                                preset_download = gr.File(label="Download audio")

                    # Voice design (VoiceDesign)
                    with gr.Tab("Voice Design (unavailable in audio.cpp)"):
                        gr.Markdown("⚠️ VoiceDesign checkpoints are not included in this Base-only audio.cpp candidate.")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=320):
                                design_name = gr.Textbox(label="Profile name", placeholder="e.g. 'Warm storyteller (designed)'")
                                design_language = gr.Dropdown(
                                    label="Language",
                                    choices=[
                                        "Auto",
                                        "English",
                                        "Spanish",
                                        "Chinese",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Italian",
                                    ],
                                    value="Auto",
                                )
                                design_instructions = gr.Textbox(
                                    label="Voice description / instructions",
                                    placeholder="e.g. A warm, friendly female voice, mid-30s, clear diction, gentle energy.",
                                    lines=4,
                                )
                                design_ref_line = gr.Textbox(
                                    label="Reference line to synthesize (this gets saved as the transcript)",
                                    value=DEFAULT_REFERENCE_LINE,
                                    lines=3,
                                )
                                design_generate_btn = gr.Button("Generate reference clip (unsupported)", variant="primary", interactive=False)
                                design_save_as_clone_btn = gr.Button("Save as reusable clone profile (unsupported)", variant="secondary", interactive=False)
                            with gr.Column(scale=1, min_width=320):
                                design_audio = gr.Audio(label="Reference audio (output, trimmable)", type="filepath", editable=True)
                                design_download = gr.File(label="Download reference audio")

                    # Voice clone (Base)
                    with gr.Tab("Voice Clone (Base)"):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=320):
                                clone_name = gr.Textbox(label="Profile name", placeholder="e.g. 'Facu - Mic Clone v1'")
                                clone_language = gr.Dropdown(
                                    label="Language",
                                    choices=[
                                        "Auto",
                                        "English",
                                        "Spanish",
                                        "Chinese",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Italian",
                                    ],
                                    value="Auto",
                                )
                                clone_ref_audio = gr.Audio(
                                    label="Reference audio (upload or record)",
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                )
                                clone_xvec_only = gr.Checkbox(
                                    label="x_vector_only_mode (no transcript needed, usually lower quality)",
                                    value=False,
                                )
                                clone_ref_text = gr.Textbox(
                                    label="Reference transcript (recommended)",
                                    placeholder="Paste the transcript of the reference audio (or leave blank if x_vector_only_mode).",
                                    lines=3,
                                )
                                clone_test_text = gr.Textbox(
                                    label="Test text (what you want to synthesize)",
                                    value="Hello! This is a voice clone test.",
                                    lines=3,
                                )
                                clone_generate_btn = gr.Button("Generate", variant="primary")
                                clone_save_btn = gr.Button("Save clone profile", variant="secondary")
                            with gr.Column(scale=1, min_width=320):
                                clone_audio = gr.Audio(label="Output audio (trimmable)", type="filepath", editable=True)
                                clone_download = gr.File(label="Download audio")

            # Library tab
            with gr.Tab("Library"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=520):
                        library_table = gr.Dataframe(
                            headers=TABLE_HEADERS,
                            datatype=["str"] * len(TABLE_HEADERS),
                            label="Saved profiles",
                            interactive=False,
                            wrap=True,
                            value=initial_profile_rows,
                        )
                        with gr.Row():
                            library_refresh_btn = gr.Button("Refresh list", variant="primary")
                            export_selected_btn = gr.Button("Export selected → ZIP", variant="secondary")
                            export_all_btn = gr.Button("Export ALL → ZIP", variant="secondary")
                        export_file = gr.File(label="Export download")
                        with gr.Row():
                            import_zip_file = gr.File(label="Import profile ZIP", file_types=[".zip"], type="filepath")
                            import_zip_btn = gr.Button("Import ZIP", variant="secondary")
                    with gr.Column(scale=1, min_width=340):
                        selected_id = gr.Textbox(label="Selected profile id", placeholder="Click a row to copy id here")
                        load_selected_btn = gr.Button("Load selected", variant="primary")
                        edit_name = gr.Textbox(label="Edit profile name", placeholder="Load a profile, then enter a new name")
                        edit_language = gr.Dropdown(
                            label="Edit language",
                            choices=["Auto", "English", "Spanish", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Italian"],
                            value="Auto",
                        )
                        edit_ref_text = gr.Textbox(label="Edit reference transcript", lines=3)
                        save_profile_edits_btn = gr.Button("Save profile changes", variant="secondary")
                        delete_selected_btn = gr.Button("Delete selected", variant="stop")
                        profile_details = gr.JSON(label="Profile details")
                        ref_preview = gr.Audio(label="Reference audio preview (if available)", type="filepath")

            # Playground tab
            with gr.Tab("Playground"):
                gr.Markdown(
                    "**Candidate capabilities:** audio.cpp Qwen3 Base supports buffered/offline clone synthesis. "
                    "The live surface below exercises microphone, VAD, cancellation, LLM turn flow, and playback scaffolding, "
                    "but it is not native incremental model PCM. Compare 0.6B vs 1.7B by switching the single resident model "
                    "under **Settings & candidate model controls**.",
                    elem_classes=["small"],
                )
                play_mode = gr.Radio(
                    ["Non-streaming (Full Quality)", "Streaming scaffolding (Buffered audio.cpp TTS)"],
                    label="Mode",
                    value="Streaming scaffolding (Buffered audio.cpp TTS)",
                )
                # --- Non-streaming mode ---
                with gr.Column(visible=False) as ns_group:
                    with gr.Row():
                        with gr.Column(scale=1, min_width=360):
                            play_profile_id = gr.Dropdown(
                                label="Pick a saved profile",
                                choices=initial_profile_ids,
                                value=initial_profile_value,
                                # Keep a stale browser selection from being
                                # rejected before the refresh callback can
                                # reconcile it with the persistent library.
                                allow_custom_value=True,
                            )
                            play_text = gr.Textbox(label="Text to synthesize", value="Hello from the playground!", lines=4)
                            play_response_format = gr.Dropdown(
                                label="Audio format",
                                choices=["wav", "mp3", "flac", "aac", "opus", "pcm"],
                                value="wav",
                            )
                            play_speed = gr.Slider(label="Speed", minimum=0.25, maximum=4.0, value=1.0, step=0.05)
                            play_seed = gr.Number(
                                label="Seed (-1 = random)",
                                value=-1,
                                precision=0,
                                minimum=-1,
                                maximum=2147483647,
                            )
                            play_generate_btn = gr.Button("🎙️ Generate (Full Quality)", variant="primary")
                        with gr.Column(scale=1, min_width=360):
                            play_audio = gr.Audio(label="Output audio (trimmable)", type="filepath", editable=True)
                            play_download = gr.File(label="Download audio")
                            play_timing = gr.Markdown(value="")
                # --- Streaming mode ---
                with gr.Column(visible=True) as s_group:
                    with gr.Row():
                        with gr.Column(scale=1, min_width=360):
                            s_voice_profile_id = gr.Dropdown(
                                label="TTS Voice Profile",
                                choices=initial_profile_ids,
                                value=initial_profile_value,
                                interactive=True,
                                allow_custom_value=True,
                            )
                            gr.Markdown(
                                "**Widget settings** (endpoint, model, API key, system prompt) are stored in your browser's local storage.",
                                elem_classes=["small"],
                            )
                        with gr.Column(scale=2, min_width=480):
                            s_streaming_widget = gr.HTML(
                                value="<div class='svwidget'><p style='color:#999;text-align:center;padding:24px;'>Select a voice profile and switch to Streaming mode to enable the live voice widget.</p></div>"
                            )

        # ------------------------------------------------------------------
        # Streaming mode widget builder
        # ------------------------------------------------------------------

        def _build_streaming_widget_html(tts_base_url: str, voice_name: str) -> str:
            """Return the HTML+JS for the streaming live-voice widget."""
            uid = "svw" + uuid.uuid4().hex[:6]
            tmpl = r'''<div class="svwidget" id="__UID__">
<style>
.__UID__-settings input, .__UID__-settings textarea { padding:6px 8px; border:1px solid #d0d0d0; border-radius:6px; font-size:0.9em; width:100%; box-sizing:border-box; }
.__UID__-settings label { display:flex; flex-direction:column; font-size:0.9em; gap:2px; }
.__UID__-viz { display:flex; align-items:center; gap:3px; height:28px; margin:8px 0; }
  .__UID__-viz span { display:inline-block; width:4px; border-radius:2px; background:#f97316; height:4px; transition:height 0.15s; }
.__UID__-viz.active span { animation: __UID__-bounce 0.6s infinite alternate; }
.__UID__-viz.active span:nth-child(1) { animation-delay:0.0s; height:6px; }
.__UID__-viz.active span:nth-child(2) { animation-delay:0.1s; height:16px; }
.__UID__-viz.active span:nth-child(3) { animation-delay:0.2s; height:24px; }
.__UID__-viz.active span:nth-child(4) { animation-delay:0.3s; height:28px; }
.__UID__-viz.active span:nth-child(5) { animation-delay:0.2s; height:24px; }
.__UID__-viz.active span:nth-child(6) { animation-delay:0.1s; height:16px; }
.__UID__-viz.active span:nth-child(7) { animation-delay:0.0s; height:6px; }
@keyframes __UID__-bounce { from { transform:scaleY(0.3); } to { transform:scaleY(1); } }
.__UID__-rviz { display:flex; align-items:flex-end; gap:2px; height:36px; margin:4px 0; }
.__UID__-rviz span { display:inline-block; width:5px; border-radius:2px 2px 0 0; background:#22c55e; height:2px; transition:height 0.06s; }
</style>

<script>
(function() {
  function registerData() {
    Alpine.data('svStreaming', function () {
      return {
        endpoint: localStorage.getItem('sv_endpoint') || 'http://localhost:8080/v1/chat/completions',
        modelName: localStorage.getItem('sv_model') || 'gemma-4-e4b',
        apiKey: localStorage.getItem('sv_api_key') || '',
        systemPrompt: localStorage.getItem('sv_system_prompt') || 'You are a helpful assistant. Keep responses concise.',
        voiceName: '__VOICE_NAME__',
        ttsBaseUrl: '__TTS_BASE_URL__',
        status: 'Ready. Click the mic to start.',
        isRecording: false,
        isProcessing: false,
        isSpeaking: false,
        transcript: [],
        historyMessages: [],
        audioContext: null,
        mediaRecorder: null,
        audioStream: null,
        audioChunks: [],
        ttfMs: null,
        ttsMs: null,
        micDevices: [],
        selectedMicId: '',
        micAnalyser: null,
        micAnimationId: null,

        init() {
          this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
          this.initMicDevices();
        },

        initMicDevices() {
          var self = this;
          navigator.mediaDevices.enumerateDevices().then(function(devices) {
            var mics = devices.filter(function(d) { return d.kind === 'audioinput'; });
            self.micDevices = mics;
            var saved = localStorage.getItem('sv_mic_id');
            if (saved && mics.some(function(m) { return m.deviceId === saved; })) {
              self.selectedMicId = saved;
            } else if (mics.length > 0) {
              self.selectedMicId = mics[0].deviceId;
            }
          });
        },

        saveMicDevice() {
          localStorage.setItem('sv_mic_id', this.selectedMicId);
        },

        startRecordViz(stream) {
          var self = this;
          try {
            var audioCtx = this.audioContext;
            var source = audioCtx.createMediaStreamSource(stream);
            var analyser = audioCtx.createAnalyser();
            analyser.fftSize = 64;
            source.connect(analyser);
            self.micAnalyser = analyser;
            var bufferLength = analyser.frequencyBinCount;
            var dataArray = new Uint8Array(bufferLength);
            var bars = [];
            for (var i = 0; i < 16; i++) {
              bars.push(document.getElementById('__UID__-rv-' + i));
            }
            function animate() {
              analyser.getByteFrequencyData(dataArray);
              for (var i = 0; i < 16; i++) {
                var idx = Math.floor(i * bufferLength / 16);
                var val = dataArray[idx] / 255;
                var h = Math.max(2, val * 34);
                if (bars[i]) bars[i].style.height = h + 'px';
              }
              self.micAnimationId = requestAnimationFrame(animate);
            }
            self.micAnimationId = requestAnimationFrame(animate);
          } catch(e) {
            // Analyser not critical; continue without viz
          }
        },

        stopRecordViz() {
          if (this.micAnimationId) {
            cancelAnimationFrame(this.micAnimationId);
            this.micAnimationId = null;
          }
          this.micAnalyser = null;
          for (var i = 0; i < 16; i++) {
            var bar = document.getElementById('__UID__-rv-' + i);
            if (bar) bar.style.height = '2px';
          }
        },

        scrollTranscript() {
          setTimeout(function() {
            var el = document.getElementById('__UID__-transcript');
            if (el) el.scrollTop = el.scrollHeight;
          }, 10);
        },

      saveSettings() {
        localStorage.setItem('sv_endpoint', this.endpoint);
        localStorage.setItem('sv_model', this.modelName);
        localStorage.setItem('sv_api_key', this.apiKey);
        localStorage.setItem('sv_system_prompt', this.systemPrompt);
        this.status = 'Settings saved.';
      },

      toggleMic() {
        if (this.isRecording) this.stopRecording();
        else this.startRecording();
      },

      async startRecording() {
        try {
          this.audioChunks = [];
          var constraints = { audio: true };
          if (this.selectedMicId) {
            constraints = { audio: { deviceId: { exact: this.selectedMicId } } };
          }
          this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
          this.startRecordViz(this.audioStream);
          this.mediaRecorder = new MediaRecorder(this.audioStream);
          var self = this;
          this.mediaRecorder.ondataavailable = function(e) { if (e.data.size > 0) self.audioChunks.push(e.data); };
          this.mediaRecorder.onstop = function() {
            self.audioStream.getTracks().forEach(function(t) { t.stop(); });
            var blob = new Blob(self.audioChunks, { type: self.audioChunks[0] ? self.audioChunks[0].type : 'audio/webm' });
            self.sendToLLM(blob);
          };
          this.mediaRecorder.start();
          this.isRecording = true;
          this.status = 'Recording... click mic again to stop.';
        } catch (err) {
          this.status = 'Mic error: ' + err.message;
        }
      },

      stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
          this.mediaRecorder.stop();
          this.isRecording = false;
          this.isProcessing = true;
          this.status = 'Sending to LLM...';
        }
        this.stopRecordViz();
      },

      sendToLLM(audioBlob) {
        var self = this;
        self.ttfMs = null;
        self.ttsMs = null;
        var reader = new FileReader();
        reader.onload = function () {
          var audioDataUri = reader.result;
          var userMsgId = Date.now();
          var messages = [{ role: 'system', content: self.systemPrompt }];
          for (var i = 0; i < self.historyMessages.length; i++) {
            var m = self.historyMessages[i];
            if (m.role === 'user' && m.audioUri) {
              messages.push({ role: 'user', content: [{ type: 'audio_url', audio_url: { url: m.audioUri } }, { type: 'text', text: '' }] });
            } else {
              messages.push({ role: m.role, content: m.content });
            }
          }
          messages.push({ role: 'user', content: [{ type: 'audio_url', audio_url: { url: audioDataUri } }, { type: 'text', text: '' }] });

          self.transcript.push({ id: userMsgId, role: 'user', content: '[Voice message]' });
          self.scrollTranscript();
          self.historyMessages.push({ role: 'user', audioUri: audioDataUri, content: '' });

          var headers = { 'Content-Type': 'application/json' };
          if (self.apiKey) headers['Authorization'] = 'Bearer ' + self.apiKey;

          var startTime = performance.now();
          fetch(self.endpoint, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
              model: self.modelName,
              messages: messages,
              stream: true,
              max_tokens: 512
            })
          }).then(function (response) {
            if (!response.ok) {
              return response.text().then(function (t) { throw new Error(response.status + ': ' + t); });
            }
            var assistantId = Date.now() + 1;
            self.transcript.push({ id: assistantId, role: 'assistant', content: '' });
            self.scrollTranscript();
            var content = '';
            var firstToken = true;
            var rdr = response.body.getReader();
            var decoder = new TextDecoder();
            var buf = '';
            function readStream() {
              rdr.read().then(function (result) {
                if (result.done) {
                  if (content.trim()) {
                    self.historyMessages.push({ role: 'assistant', content: content });
                    var lastMsg = self.transcript[self.transcript.length - 1];
                    if (lastMsg && lastMsg.id === assistantId) {
                      lastMsg.content = content;
                      lastMsg.timing = self.ttfMs ? 'TTFB: ' + self.ttfMs + 'ms' : '';
                      self.transcript = self.transcript.slice();
                    }
                    self.status = 'Synthesizing speech...';
                    self.synthesizeSpeech(content);
                  }
                  return;
                }
                buf += decoder.decode(result.value, { stream: true });
                var lines = buf.split('\n');
                buf = lines.pop() || '';
                for (var j = 0; j < lines.length; j++) {
                  var line = lines[j];
                  if (!line.startsWith('data: ')) continue;
                  var data = line.slice(6).trim();
                  if (!data || data === '[DONE]') continue;
                  try {
                    var json = JSON.parse(data);
                    var delta = json.choices && json.choices[0] && json.choices[0].delta ? (json.choices[0].delta.content || '') : '';
                    if (delta) {
                      if (firstToken) { self.ttfMs = Math.round(performance.now() - startTime); firstToken = false; }
                      content += delta;
                      var lastMsg = self.transcript[self.transcript.length - 1];
                      if (lastMsg && lastMsg.id === assistantId) {
                        lastMsg.content = content;
                        self.transcript = self.transcript.slice();
                        self.scrollTranscript();
                      }
                    }
                  } catch(e) {}
                }
                readStream();
              }).catch(function (err) {
                self.status = 'Stream error: ' + err.message;
                self.isProcessing = false;
              });
            }
            readStream();
          }).catch(function (err) {
            self.status = 'LLM error: ' + err.message;
            self.transcript.push({ id: Date.now(), role: 'assistant', content: 'Error: ' + err.message });
            self.isProcessing = false;
          });
        };
        reader.readAsDataURL(audioBlob);
      },

      synthesizeSpeech(text) {
        var self = this;
        self.isSpeaking = true;
        var ttsStart = performance.now();
        var url = self.ttsBaseUrl + '/v1/audio/voice-clone/stream';
        fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input: text,
            response_format: 'pcm',
            task_type: 'Base',
            voice: self.voiceName,
            language: 'Auto',
            stream: true
          })
        }).then(function (response) {
          if (!response.ok) {
            return response.text().then(function (t) { throw new Error(response.status + ': ' + t); });
          }
          var rdr = response.body.getReader();
          var decoder = new TextDecoder();
          var buf = new Uint8Array(0);
          var sampleRate = 24000;
          function readTts() {
            rdr.read().then(function (result) {
              if (result.done) {
                self.isSpeaking = false;
                self.ttsMs = Math.round(performance.now() - ttsStart);
                self.isProcessing = false;
                self.status = 'Ready. Click the mic to continue.';
                return;
              }
              var newBuf = new Uint8Array(buf.length + result.value.length);
              newBuf.set(buf);
              newBuf.set(result.value, buf.length);
              buf = newBuf;
              var offset = 0;
              while (offset + 8 <= buf.length) {
                var jsonLen = new DataView(buf.buffer, offset, 4).getUint32(0, true);
                if (offset + 4 + jsonLen + 4 > buf.length) break;
                var jsonStr = decoder.decode(buf.slice(offset + 4, offset + 4 + jsonLen));
                try { var meta = JSON.parse(jsonStr); } catch(e) { offset++; continue; }
                var audioLen = new DataView(buf.buffer, offset + 4 + jsonLen, 4).getUint32(0, true);
                if (offset + 4 + jsonLen + 4 + audioLen > buf.length) break;
                var audioData = buf.slice(offset + 4 + jsonLen + 4, offset + 4 + jsonLen + 4 + audioLen);
                offset += 4 + jsonLen + 4 + audioLen;
                if (audioData.length > 0) {
                  self.playPcm(audioData, sampleRate);
                }
              }
              buf = buf.slice(offset);
              readTts();
            }).catch(function (err) {
              self.isSpeaking = false;
              self.status = 'TTS stream error: ' + err.message;
              self.isProcessing = false;
            });
          }
          readTts();
        }).catch(function (err) {
          self.isSpeaking = false;
          self.status = 'TTS error: ' + err.message;
          self.isProcessing = false;
        });
      },

      playPcm(pcmData, sampleRate) {
        var float32Data = new Float32Array(pcmData.length / 2);
        for (var i = 0; i < float32Data.length; i++) {
          var sample = (pcmData[i * 2] | (pcmData[i * 2 + 1] << 8));
          float32Data[i] = sample / 32768.0;
        }
        var audioBuf = this.audioContext.createBuffer(1, float32Data.length, sampleRate);
        audioBuf.getChannelData(0).set(float32Data);
        var source = this.audioContext.createBufferSource();
        source.buffer = audioBuf;
        source.connect(this.audioContext.destination);
        source.start();
      },

      clearTranscript() {
        this.transcript = [];
        this.historyMessages = [];
        this.status = 'Conversation cleared.';
        this.ttfMs = null;
        this.ttsMs = null;
      }
    };
  }
})();

if (typeof Alpine === 'undefined') {
  if (typeof window._svAlpineLoading === 'undefined') {
    window._svAlpineLoading = true;
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js';
    document.head.appendChild(s);
    s.onload = function() { Alpine.data('svStreaming', function() { return {}; }); location.reload(); };
  }
} else {
  registerData();
}
</script>

<div x-data="svStreaming()" x-init="init()" style="min-height:320px">
  <details>
    <summary style="cursor:pointer;font-weight:600;">LLM Settings (stored locally)</summary>
    <div class="__UID__-settings" style="display:grid;gap:8px;padding:8px 0;">
      <label>Multimodal LLM Endpoint:
        <input type="text" x-model="endpoint" @change.debounce="saveSettings()" placeholder="http://localhost:8080/v1/chat/completions">
      </label>
      <label>Model Name:
        <input type="text" x-model="modelName" @change.debounce="saveSettings()" placeholder="gemma-4-e4b">
      </label>
      <label>API Key (optional):
        <input type="password" x-model="apiKey" @change.debounce="saveSettings()" placeholder="sk-...">
      </label>
      <label>System Prompt:
        <textarea x-model="systemPrompt" @change.debounce="saveSettings()" rows="3" placeholder="You are a helpful assistant."></textarea>
      </label>
    </div>
  </details>

  <!-- Microphone card -->
  <div class="sv-card">
    <div class="sv-card-header">
      <span class="sv-card-title">🎤 Microphone</span>
      <span class="sv-card-badge" :class="{ rec: isRecording }" x-show="isRecording || isProcessing" x-text="isRecording ? 'REC' : 'BUSY'"></span>
    </div>
    <div class="sv-card-body">
      <select class="sv-mic-select" x-model="selectedMicId" @change="saveMicDevice()" :disabled="isRecording">
        <template x-for="d in micDevices" :key="d.deviceId">
          <option x-bind:value="d.deviceId" x-text="d.label || 'Microphone (' + d.deviceId.slice(0,8) + '...)'"></option>
        </template>
      </select>
      <div class="sv-mic-row">
        <button class="sv-mic-btn" :class="{ recording: isRecording }" @click="toggleMic()" :disabled="isProcessing">
          <span x-text="isRecording ? '⏹️ Stop' : (isProcessing ? '⏳ Processing...' : '🎤 Push to Talk')">🎤 Push to Talk</span>
        </button>
      </div>
      <div class="__UID__-rviz">
        <span id="__UID__-rv-0"></span><span id="__UID__-rv-1"></span><span id="__UID__-rv-2"></span><span id="__UID__-rv-3"></span>
        <span id="__UID__-rv-4"></span><span id="__UID__-rv-5"></span><span id="__UID__-rv-6"></span><span id="__UID__-rv-7"></span>
        <span id="__UID__-rv-8"></span><span id="__UID__-rv-9"></span><span id="__UID__-rv-10"></span><span id="__UID__-rv-11"></span>
        <span id="__UID__-rv-12"></span><span id="__UID__-rv-13"></span><span id="__UID__-rv-14"></span><span id="__UID__-rv-15"></span>
      </div>
      <div class="sv-card-status"><span x-text="status"></span></div>
    </div>
  </div>

  <!-- Conversation card -->
  <div class="sv-card">
    <div class="sv-card-header">
      <span class="sv-card-title">💬 Conversation</span>
      <button class="sv-clr-btn" @click="clearTranscript()" x-show="transcript.length > 0">🗑️ Clear</button>
    </div>
    <div class="sv-card-body sv-card-body-nopad">
      <div class="sv-transcript" id="__UID__-transcript" x-ref="transcript">
        <template x-for="msg in transcript" :key="msg.id">
          <div class="sv-msg" :class="msg.role">
            <div class="role" x-text="msg.role === 'user' ? '🎤 You' : '🤖 Assistant'"></div>
            <div class="content" x-text="msg.content"></div>
            <template x-if="msg.timing">
              <div class="sv-timing" x-text="msg.timing"></div>
            </template>
          </div>
        </template>
        <template x-if="transcript.length === 0">
          <div style="color:#999;text-align:center;padding:24px;">Click the mic button and speak to start a conversation.</div>
        </template>
      </div>
    </div>
  </div>

  <!-- Voice Output card -->
  <div class="sv-card">
    <div class="sv-card-header">
      <span class="sv-card-title">🔊 Voice Output</span>
      <span class="sv-card-badge speaking" x-show="isSpeaking">Speaking</span>
    </div>
    <div class="sv-card-body">
      <div class="__UID__-viz" :class="{ active: isSpeaking }">
        <span></span><span></span><span></span><span></span><span></span><span></span><span></span>
      </div>
      <div class="sv-metrics">
        <template x-if="ttfMs">
          <span>LLM TTFB: <b x-text="ttfMs"></b>ms</span>
        </template>
        <template x-if="ttsMs">
          <span>TTS: <b x-text="ttsMs"></b>ms</span>
        </template>
      </div>
    </div>
  </div>
</div>'''
            return (tmpl
                .replace('__TTS_BASE_URL__', tts_base_url)
                .replace('__VOICE_NAME__', voice_name)
                .replace('__UID__', uid)
                .replace('__UID__', uid)) 

        def _build_streaming_widget_html(
            tts_base_url: str,
            voice_name: str,
            tts_voice: str,
            profile_label: str,
            language: str = "Auto",
        ) -> str:
            """Return a self-contained streaming live-voice widget."""
            uid = "svw" + uuid.uuid4().hex[:6]
            config = json.dumps(
                {
                    "ttsBaseUrl": normalize_base_url(tts_base_url),
                    "voiceName": voice_name or "Vivian",
                    "ttsVoice": tts_voice or voice_name or "Vivian",
                    # Browser requests stay on the Gradio host (8891) and are
                    # proxied to the supervisor.  Do not make the widget fetch
                    # the candidate API port directly; that triggers CORS.
                    "proxyBaseUrl": "/api/audio-cpp",
                    "profileLabel": profile_label or voice_name or "Selected profile",
                    "language": language or "Auto",
                    "sampleRate": 24000,
                }
            )
            tmpl = r'''<div class="svwidget" id="__UID__">
<style>
#__UID__ {
  --sv-border: rgba(15, 23, 42, 0.14);
  --sv-bg: #ffffff;
  --sv-muted: rgba(15, 23, 42, 0.66);
  --sv-text: #111827;
  --sv-accent: #f97316;
  --sv-accent-soft: rgba(249, 115, 22, 0.10);
  --sv-ok: #16a34a;
  --sv-danger: #dc2626;
  color: var(--sv-text);
  display: block;
  font-family: inherit;
}
#__UID__ * { box-sizing: border-box; }
#__UID__ .sv-settings {
  border: 1px solid var(--sv-border);
  border-radius: 8px;
  background: var(--sv-bg);
  margin: 0 0 12px;
}
#__UID__ .sv-settings summary {
  cursor: pointer;
  font-weight: 600;
  padding: 10px 12px;
}
#__UID__ .sv-settings-grid {
  border-top: 1px solid var(--sv-border);
  display: grid;
  gap: 10px;
  padding: 12px;
}
#__UID__ .sv-settings-title {
  color: var(--sv-text);
  font-size: 0.88rem;
  font-weight: 700;
  margin: 2px 0 0;
}
#__UID__ .sv-settings-row {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(3, minmax(120px, 1fr));
}
#__UID__ .sv-field {
  color: var(--sv-muted);
  display: flex;
  flex-direction: column;
  font-size: 0.9rem;
  gap: 5px;
}
#__UID__ .sv-input,
#__UID__ .sv-select,
#__UID__ .sv-textarea {
  background: #fff;
  border: 1px solid var(--sv-border);
  border-radius: 8px;
  color: var(--sv-text);
  font: inherit;
  min-height: 42px;
  padding: 8px 10px;
  width: 100%;
}
#__UID__ .sv-textarea {
  min-height: 82px;
  resize: vertical;
}
#__UID__ .sv-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr);
}
#__UID__ .sv-panel {
  background: var(--sv-bg);
  border: 1px solid var(--sv-border);
  border-radius: 8px;
  min-width: 0;
  overflow: hidden;
}
#__UID__ .sv-panel-wide {
  grid-column: 1 / -1;
}
#__UID__ .sv-panel-head {
  align-items: center;
  border-bottom: 1px solid var(--sv-border);
  display: flex;
  gap: 10px;
  justify-content: space-between;
  min-height: 42px;
  padding: 9px 12px;
}
#__UID__ .sv-label {
  color: var(--sv-muted);
  font-size: 0.86rem;
  font-weight: 600;
}
#__UID__ .sv-panel-body {
  display: grid;
  gap: 10px;
  padding: 12px;
}
#__UID__ .sv-row {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
#__UID__ .sv-button {
  align-items: center;
  border: 1px solid var(--sv-border);
  border-radius: 8px;
  cursor: pointer;
  display: inline-flex;
  font: inherit;
  font-weight: 600;
  justify-content: center;
  min-height: 40px;
  padding: 8px 14px;
}
#__UID__ .sv-button-primary {
  background: var(--sv-accent);
  border-color: var(--sv-accent);
  color: #fff;
}
#__UID__ .sv-button-secondary {
  background: #fff;
  color: var(--sv-text);
}
#__UID__ .sv-button-danger {
  background: var(--sv-danger);
  border-color: var(--sv-danger);
  color: #fff;
}
#__UID__ .sv-button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
#__UID__ .sv-badge {
  background: rgba(15, 23, 42, 0.06);
  border: 1px solid var(--sv-border);
  border-radius: 999px;
  color: var(--sv-muted);
  font-size: 0.75rem;
  font-weight: 700;
  padding: 3px 9px;
}
#__UID__ .sv-badge-recording {
  background: rgba(220, 38, 38, 0.10);
  border-color: rgba(220, 38, 38, 0.22);
  color: var(--sv-danger);
}
#__UID__ .sv-badge-speaking {
  background: var(--sv-accent-soft);
  border-color: rgba(249, 115, 22, 0.25);
  color: #c2410c;
}
#__UID__ .sv-status {
  color: var(--sv-muted);
  font-size: 0.88rem;
  min-height: 22px;
}
#__UID__ .sv-meter {
  align-items: flex-end;
  display: flex;
  gap: 3px;
  height: 36px;
  padding: 4px 0;
}
#__UID__ .sv-meter span {
  background: var(--sv-accent);
  border-radius: 2px 2px 0 0;
  display: inline-block;
  height: 3px;
  transition: height 0.06s ease;
  width: 5px;
}
#__UID__ .sv-wave {
  align-items: center;
  display: flex;
  gap: 4px;
  height: 36px;
}
#__UID__ .sv-wave span {
  background: var(--sv-accent);
  border-radius: 999px;
  display: inline-block;
  height: 5px;
  opacity: 0.45;
  width: 5px;
}
#__UID__ .sv-wave-active span {
  animation: __UID__-wave 0.7s ease-in-out infinite alternate;
}
#__UID__ .sv-wave-active span:nth-child(2) { animation-delay: 0.08s; }
#__UID__ .sv-wave-active span:nth-child(3) { animation-delay: 0.16s; }
#__UID__ .sv-wave-active span:nth-child(4) { animation-delay: 0.24s; }
#__UID__ .sv-wave-active span:nth-child(5) { animation-delay: 0.16s; }
#__UID__ .sv-wave-active span:nth-child(6) { animation-delay: 0.08s; }
@keyframes __UID__-wave {
  from { height: 5px; opacity: 0.45; }
  to { height: 30px; opacity: 1; }
}
#__UID__ .sv-audio {
  min-height: 48px;
}
#__UID__ .sv-audio audio {
  display: block;
  width: 100%;
}
#__UID__ .sv-empty-audio {
  align-items: center;
  border: 1px dashed var(--sv-border);
  border-radius: 8px;
  color: var(--sv-muted);
  display: flex;
  min-height: 48px;
  padding: 0 12px;
}
#__UID__ .sv-transcript {
  background: #fafafa;
  border: 1px solid var(--sv-border);
  border-radius: 8px;
  max-height: 310px;
  min-height: 168px;
  overflow-y: auto;
  padding: 10px;
}
#__UID__ .sv-message {
  border: 1px solid var(--sv-border);
  border-radius: 8px;
  margin-bottom: 8px;
  padding: 8px 10px;
}
#__UID__ .sv-message-user {
  background: rgba(249, 115, 22, 0.08);
}
#__UID__ .sv-message-assistant {
  background: #fff;
}
#__UID__ .sv-message-role {
  color: var(--sv-muted);
  font-size: 0.78rem;
  font-weight: 700;
  margin-bottom: 4px;
}
#__UID__ .sv-message-text {
  white-space: pre-wrap;
  word-break: break-word;
}
#__UID__ .sv-message-timing,
#__UID__ .sv-metrics {
  color: var(--sv-muted);
  font-size: 0.82rem;
}
#__UID__ .sv-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
#__UID__ .sv-hidden {
  display: none !important;
}
@media (max-width: 760px) {
  #__UID__ .sv-grid {
    grid-template-columns: 1fr;
  }
  #__UID__ .sv-settings-row {
    grid-template-columns: 1fr;
  }
}
</style>

<details class="sv-settings">
  <summary>Live scaffolding settings (TTS output is buffered)</summary>
  <div class="sv-settings-grid">
    <div class="sv-settings-title">llama.cpp chat</div>
    <label class="sv-field">llama.cpp endpoint
      <input class="sv-input" data-role="endpoint" type="text" placeholder="http://127.0.0.1:1234">
    </label>
    <label class="sv-field">Model name
      <input class="sv-input" data-role="model" type="text" placeholder="gemma-4-e4b">
    </label>
    <label class="sv-field">API key
      <input class="sv-input" data-role="api-key" type="password" placeholder="Optional">
    </label>
    <label class="sv-field">System prompt optional
      <textarea class="sv-textarea" data-role="system-prompt" rows="3" placeholder="Leave empty to use only the server-side prompt."></textarea>
    </label>
    <div class="sv-row">
      <label class="sv-field"><input data-role="remember-api-key" type="checkbox"> Remember API key on this browser</label>
      <button class="sv-button sv-button-secondary" data-action="save-llm-settings" type="button">Save LLM settings</button>
      <span class="sv-status" data-role="llm-save-status"></span>
    </div>
    <div class="sv-settings-title">Mic thresholds</div>
    <div class="sv-settings-row">
      <label class="sv-field">Speech threshold
        <input class="sv-input" data-role="speech-threshold" type="number" min="0.001" max="1" step="0.001">
      </label>
      <label class="sv-field">Start hold ms
        <input class="sv-input" data-role="start-hold" type="number" min="0" max="3000" step="25">
      </label>
      <label class="sv-field">Silence send delay ms
        <input class="sv-input" data-role="silence-delay" type="number" min="100" max="5000" step="50">
      </label>
      <label class="sv-field">Minimum utterance ms
        <input class="sv-input" data-role="min-utterance" type="number" min="100" max="5000" step="50">
      </label>
      <label class="sv-field">Maximum utterance sec
        <input class="sv-input" data-role="max-utterance" type="number" min="2" max="180" step="1">
      </label>
      <label class="sv-field">Pre-roll ms
        <input class="sv-input" data-role="pre-roll" type="number" min="0" max="2000" step="25">
      </label>
    </div>
    <div class="sv-settings-title">Progressive PCM diagnostics</div>
    <div class="sv-settings-row">
      <label class="sv-field">Phrase minimum characters
        <input class="sv-input" data-role="phrase-min" type="number" min="8" max="240" step="1">
      </label>
      <label class="sv-field">Phrase maximum characters
        <input class="sv-input" data-role="phrase-max" type="number" min="24" max="400" step="1">
      </label>
      <label class="sv-field">Idle flush ms
        <input class="sv-input" data-role="phrase-idle" type="number" min="100" max="3000" step="25">
      </label>
    </div>
    <p class="sv-status">Buffered phrase PCM: each phrase completes in audio.cpp before playback. Native model-level incremental PCM is unavailable in this build.</p>
  </div>
</details>

<div class="sv-grid">
  <section class="sv-panel">
    <div class="sv-panel-head">
      <span class="sv-label">User microphone</span>
      <span class="sv-badge sv-badge-recording sv-hidden" data-role="mic-badge">Listening</span>
    </div>
    <div class="sv-panel-body">
      <label class="sv-field">Input device
        <select class="sv-select" data-role="mic-select"></select>
      </label>
      <div class="sv-row">
        <button class="sv-button sv-button-secondary" data-action="refresh-mics" type="button">Refresh microphones</button>
        <button class="sv-button sv-button-primary" data-action="toggle-record" type="button">Start live mic</button>
      </div>
      <div class="sv-meter" data-role="mic-meter">
        <span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span>
      </div>
      <div class="sv-status" data-role="status">Ready. Choose a microphone and start live testing.</div>
      <div class="sv-audio">
        <audio class="sv-hidden" data-role="user-audio" controls></audio>
        <div class="sv-empty-audio" data-role="user-empty">Last detected utterance will appear here.</div>
      </div>
    </div>
  </section>

  <section class="sv-panel">
    <div class="sv-panel-head">
      <span class="sv-label">AI voice output</span>
      <span class="sv-badge sv-badge-speaking sv-hidden" data-role="voice-badge">Speaking</span>
    </div>
    <div class="sv-panel-body">
      <div class="sv-status" data-role="voice-name"></div>
      <div class="sv-wave" data-role="voice-wave"><span></span><span></span><span></span><span></span><span></span><span></span></div>
      <div class="sv-metrics">
        <span>LLM TTFB: <b data-role="ttf">--</b></span>
        <span>TTS: <b data-role="tts">--</b></span>
      </div>
      <div class="sv-audio">
        <audio class="sv-hidden" data-role="ai-audio" controls></audio>
        <div class="sv-empty-audio" data-role="ai-empty">AI response audio will appear here.</div>
      </div>
    </div>
  </section>

  <section class="sv-panel sv-panel-wide">
    <div class="sv-panel-head">
      <span class="sv-label">Dialogue</span>
      <button class="sv-button sv-button-secondary" data-action="clear" type="button">Clear</button>
    </div>
    <div class="sv-panel-body">
      <div class="sv-transcript" data-role="transcript">
        <div class="sv-status">Record a message to start the conversation.</div>
      </div>
    </div>
  </section>
</div>

<script type="text/plain" id="__UID__-boot">
(function() {
  var root = document.getElementById('__UID__');
  if (!root || root.dataset.bound === '1') return;
  root.dataset.bound = '1';

  var config = __CONFIG__;
  var refs = {};
  [
    'endpoint', 'model', 'api-key', 'system-prompt', 'remember-api-key', 'save-llm-settings', 'llm-save-status', 'mic-select', 'mic-badge',
    'toggle-record', 'refresh-mics', 'mic-meter', 'status', 'user-audio',
    'user-empty', 'voice-badge', 'voice-name', 'voice-wave', 'ttf', 'tts',
    'ai-audio', 'ai-empty', 'transcript', 'clear', 'speech-threshold',
    'start-hold', 'silence-delay', 'min-utterance', 'max-utterance', 'pre-roll', 'phrase-min', 'phrase-max', 'phrase-idle'
  ].forEach(function(name) {
    refs[name] = root.querySelector('[data-role="' + name + '"], [data-action="' + name + '"]');
  });

  var state = {
    audioContext: null,
    mediaRecorder: null,
    audioStream: null,
    micSource: null,
    analyser: null,
    vadFrame: null,
    vadData: null,
    activeRequest: null,
    rollingChunks: [],
    segmentChunks: [],
    segmentStartedAt: 0,
    speechStartedAt: null,
    silenceStartedAt: null,
    isLive: false,
    isSegmenting: false,
    isFinalizingSegment: false,
    isProcessing: false,
    isSpeaking: false,
    historyMessages: [],
    transcript: [],
    userAudioUrl: null,
    aiAudioUrl: null,
    ttfMs: null,
    ttsMs: null,
    phraseText: '',
    phraseQueue: [],
    phrasePumping: false,
    llmFinished: false,
    playbackAt: 0,
    pcmChunks: [],
    turnId: 0,
    phraseIdleTimer: null
  };

  function localGet(key, fallback) {
    var value = localStorage.getItem(key);
    return value === null ? fallback : value;
  }

  function localSet(key, value) {
    localStorage.setItem(key, value || '');
  }

  function numberFromInput(name, fallback, min, max) {
    var raw = parseFloat(refs[name].value);
    if (!isFinite(raw)) raw = fallback;
    raw = Math.max(min, Math.min(max, raw));
    refs[name].value = String(raw);
    return raw;
  }

  function getVadSettings() {
    return {
      threshold: numberFromInput('speech-threshold', 0.025, 0.001, 1),
      startHoldMs: numberFromInput('start-hold', 200, 0, 3000),
      silenceDelayMs: numberFromInput('silence-delay', 900, 100, 5000),
      minUtteranceMs: numberFromInput('min-utterance', 400, 100, 5000),
      maxUtteranceMs: numberFromInput('max-utterance', 30, 2, 180) * 1000,
      preRollMs: numberFromInput('pre-roll', 250, 0, 2000)
    };
  }

  function getPhraseSettings() {
    var minimum = numberFromInput('phrase-min', 24, 8, 240);
    var maximum = Math.max(minimum, numberFromInput('phrase-max', 180, 24, 400));
    refs['phrase-max'].value = String(maximum);
    return { minimum: minimum, maximum: maximum, idleMs: numberFromInput('phrase-idle', 450, 100, 3000) };
  }

  function saveVadSettings() {
    ['speech-threshold', 'start-hold', 'silence-delay', 'min-utterance', 'max-utterance', 'pre-roll'].forEach(function(name) {
      localSet('sv_vad_' + name, refs[name].value);
    });
  }

  function loadVadSettings() {
    refs['speech-threshold'].value = localGet('sv_vad_speech-threshold', '0.025');
    refs['start-hold'].value = localGet('sv_vad_start-hold', '200');
    refs['silence-delay'].value = localGet('sv_vad_silence-delay', '900');
    refs['min-utterance'].value = localGet('sv_vad_min-utterance', '400');
    refs['max-utterance'].value = localGet('sv_vad_max-utterance', '30');
    refs['pre-roll'].value = localGet('sv_vad_pre-roll', '250');
    refs['phrase-min'].value = localGet('sv_phrase-min', '24');
    refs['phrase-max'].value = localGet('sv_phrase-max', '180');
    refs['phrase-idle'].value = localGet('sv_phrase-idle', '450');
  }

  function setStatus(message) {
    refs.status.textContent = message;
  }

  function setProcessing(active) {
    state.isProcessing = active;
    refs['refresh-mics'].disabled = active || state.isLive;
  }

  function setLive(active) {
    state.isLive = active;
    refs['mic-badge'].classList.toggle('sv-hidden', !active);
    refs['mic-badge'].textContent = active ? 'Listening' : 'Idle';
    refs['toggle-record'].textContent = active ? 'Stop live mic' : 'Start live mic';
    refs['toggle-record'].classList.toggle('sv-button-danger', active);
    refs['toggle-record'].classList.toggle('sv-button-primary', !active);
    refs['refresh-mics'].disabled = active || state.isProcessing;
  }

  function setSpeaking(active) {
    state.isSpeaking = active;
    refs['voice-badge'].classList.toggle('sv-hidden', !active);
    refs['voice-wave'].classList.toggle('sv-wave-active', active);
  }

  function resetMeter() {
    Array.prototype.forEach.call(refs['mic-meter'].querySelectorAll('span'), function(bar) {
      bar.style.height = '3px';
    });
  }

  function updateMeter(level) {
    var bars = refs['mic-meter'].querySelectorAll('span');
    Array.prototype.forEach.call(bars, function(bar, index) {
      var taper = 0.55 + (index / Math.max(1, bars.length - 1)) * 0.45;
      var h = Math.max(3, Math.min(34, level * 520 * taper));
      bar.style.height = h + 'px';
    });
  }

  function stopVadLoop() {
    if (state.vadFrame) {
      cancelAnimationFrame(state.vadFrame);
      state.vadFrame = null;
    }
    if (state.micSource) {
      try { state.micSource.disconnect(); } catch (err) {}
      state.micSource = null;
    }
    state.analyser = null;
    state.vadData = null;
    resetMeter();
  }

  function startVadLoop(stream) {
    stopVadLoop();
    try {
      var ctx = getAudioContext();
      var source = ctx.createMediaStreamSource(stream);
      var analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      state.micSource = source;
      state.analyser = analyser;
      state.vadData = new Uint8Array(analyser.fftSize);
      function tick(now) {
        analyser.getByteTimeDomainData(state.vadData);
        var sum = 0;
        for (var i = 0; i < state.vadData.length; i += 1) {
          var sample = (state.vadData[i] - 128) / 128;
          sum += sample * sample;
        }
        var rms = Math.sqrt(sum / state.vadData.length);
        updateMeter(rms);
        handleVadLevel(rms, now || performance.now());
        state.vadFrame = requestAnimationFrame(tick);
      }
      state.vadFrame = requestAnimationFrame(tick);
    } catch (err) {
      resetMeter();
    }
  }

  function getAudioContext() {
    if (!state.audioContext) {
      state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (state.audioContext.state === 'suspended') {
      state.audioContext.resume();
    }
    return state.audioContext;
  }

  function renderTranscript() {
    refs.transcript.textContent = '';
    if (!state.transcript.length) {
      var empty = document.createElement('div');
      empty.className = 'sv-status';
      empty.textContent = 'Record a message to start the conversation.';
      refs.transcript.appendChild(empty);
      return;
    }
    state.transcript.forEach(function(msg) {
      var wrap = document.createElement('div');
      wrap.className = 'sv-message sv-message-' + msg.role;
      var role = document.createElement('div');
      role.className = 'sv-message-role';
      role.textContent = msg.role === 'user' ? 'You' : 'Assistant';
      var text = document.createElement('div');
      text.className = 'sv-message-text';
      text.textContent = msg.content || '';
      wrap.appendChild(role);
      wrap.appendChild(text);
      if (msg.timing) {
        var timing = document.createElement('div');
        timing.className = 'sv-message-timing';
        timing.textContent = msg.timing;
        wrap.appendChild(timing);
      }
      refs.transcript.appendChild(wrap);
    });
    refs.transcript.scrollTop = refs.transcript.scrollHeight;
  }

  function addTranscript(role, content, timing) {
    var msg = { id: Date.now() + Math.random(), role: role, content: content || '', timing: timing || '' };
    state.transcript.push(msg);
    renderTranscript();
    return msg;
  }

  function showAudio(audioEl, emptyEl, url) {
    audioEl.src = url;
    audioEl.classList.remove('sv-hidden');
    emptyEl.classList.add('sv-hidden');
  }

  async function refreshMics(requestPermission) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices || !navigator.mediaDevices.getUserMedia) {
      refs['mic-select'].innerHTML = '<option value="">No browser microphone API</option>';
      setStatus('Browser microphone access is unavailable on this page.');
      return;
    }
    if (requestPermission) {
      try {
        var unlockStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        unlockStream.getTracks().forEach(function(track) { track.stop(); });
      } catch (err) {
        setStatus('Microphone permission was not granted: ' + err.message);
      }
    }
    var devices = await navigator.mediaDevices.enumerateDevices();
    var mics = devices.filter(function(device) { return device.kind === 'audioinput'; });
    refs['mic-select'].textContent = '';
    if (!mics.length) {
      var none = document.createElement('option');
      none.value = '';
      none.textContent = 'No microphones found';
      refs['mic-select'].appendChild(none);
      return;
    }
    var saved = localStorage.getItem('sv_mic_id') || '';
    var savedExists = mics.some(function(device) { return device.deviceId === saved; });
    if (saved && !savedExists) {
      localStorage.removeItem('sv_mic_id');
      saved = '';
    }
    var systemDefault = document.createElement('option');
    systemDefault.value = '';
    systemDefault.textContent = 'System default input';
    refs['mic-select'].appendChild(systemDefault);
    mics.forEach(function(device, index) {
      var option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || ('Microphone ' + (index + 1));
      refs['mic-select'].appendChild(option);
    });
    refs['mic-select'].value = saved || '';
    if (!requestPermission && mics.some(function(device) { return !device.label; })) {
      setStatus('Choose Refresh microphones or Start live mic to reveal browser-approved device names.');
    }
  }

  function chooseMimeType() {
    var candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/wav', 'audio/mp4'];
    for (var i = 0; i < candidates.length; i += 1) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(candidates[i])) {
        return candidates[i];
      }
    }
    return '';
  }

  async function getMicStream() {
    var selected = refs['mic-select'].value;
    if (selected) {
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: {
            deviceId: { exact: selected },
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
      } catch (err) {
        localStorage.removeItem('sv_mic_id');
        refs['mic-select'].value = '';
        setStatus('Saved microphone was unavailable. Falling back to the default input.');
      }
    }
    return await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });
  }

  function pruneRollingChunks(now, preRollMs) {
    var keepAfter = now - Math.max(preRollMs + 800, 1500);
    state.rollingChunks = state.rollingChunks.filter(function(item) { return item.time >= keepAfter; });
  }

  function onRecorderData(event) {
    if (!event.data || event.data.size <= 0) return;
    var now = performance.now();
    var item = { blob: event.data, time: now };
    state.rollingChunks.push(item);
    if (state.isSegmenting || state.isFinalizingSegment) {
      state.segmentChunks.push(event.data);
    }
    pruneRollingChunks(now, getVadSettings().preRollMs);
  }

  function beginSpeechSegment(now, settings) {
    if (state.isSegmenting || state.isFinalizingSegment || state.isProcessing) return;
    state.isSegmenting = true;
    state.segmentStartedAt = now;
    state.silenceStartedAt = null;
    state.segmentChunks = state.rollingChunks
      .filter(function(item) { return item.time >= now - settings.preRollMs; })
      .map(function(item) { return item.blob; });
    refs['mic-badge'].textContent = 'Speech';
    setStatus('Speech detected. The turn will send after silence.');
  }

  function discardSpeechSegment(message) {
    state.isSegmenting = false;
    state.isFinalizingSegment = false;
    state.segmentChunks = [];
    state.speechStartedAt = null;
    state.silenceStartedAt = null;
    refs['mic-badge'].textContent = state.isLive ? 'Listening' : 'Idle';
    if (message) setStatus(message);
  }

  function finalizeSpeechSegment() {
    if (!state.isSegmenting || state.isFinalizingSegment) return;
    state.isFinalizingSegment = true;
    setProcessing(true);
    refs['mic-badge'].textContent = 'Sending';
    setStatus('Sending detected speech to llama.cpp.');
    try {
      if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
        state.mediaRecorder.requestData();
      }
    } catch (err) {}
    window.setTimeout(function() {
      var chunks = state.segmentChunks.slice();
      var mime = chunks[0] ? chunks[0].type : 'audio/webm';
      discardSpeechSegment('');
      if (!chunks.length) {
        setProcessing(false);
        setStatus(state.isLive ? 'Listening. Speak when ready.' : 'Ready.');
        return;
      }
      var blob = new Blob(chunks, { type: mime || 'audio/webm' });
      if (state.userAudioUrl) URL.revokeObjectURL(state.userAudioUrl);
      state.userAudioUrl = URL.createObjectURL(blob);
      showAudio(refs['user-audio'], refs['user-empty'], state.userAudioUrl);
      sendToLLM(blob);
    }, 160);
  }

  function handleVadLevel(level, now) {
    if (!state.isLive || state.isProcessing || state.isSpeaking || state.isFinalizingSegment) {
      state.speechStartedAt = null;
      state.silenceStartedAt = null;
      return;
    }
    var settings = getVadSettings();
    pruneRollingChunks(now, settings.preRollMs);
    var isSpeech = level >= settings.threshold;
    if (!state.isSegmenting) {
      if (isSpeech) {
        if (state.speechStartedAt === null) state.speechStartedAt = now;
        if (now - state.speechStartedAt >= settings.startHoldMs) {
          beginSpeechSegment(now, settings);
        }
      } else {
        state.speechStartedAt = null;
      }
      return;
    }

    var duration = now - state.segmentStartedAt;
    if (duration >= settings.maxUtteranceMs) {
      finalizeSpeechSegment();
      return;
    }
    if (isSpeech) {
      state.silenceStartedAt = null;
      return;
    }
    if (state.silenceStartedAt === null) state.silenceStartedAt = now;
    if (now - state.silenceStartedAt >= settings.silenceDelayMs) {
      if (duration >= settings.minUtteranceMs) {
        finalizeSpeechSegment();
      } else {
        discardSpeechSegment('Ignored a very short sound. Listening.');
      }
    }
  }

  async function startLiveMic() {
    try {
      getAudioContext();
      state.rollingChunks = [];
      state.segmentChunks = [];
      state.audioStream = await getMicStream();
      await refreshMics(false);
      startVadLoop(state.audioStream);
      var mimeType = chooseMimeType();
      state.mediaRecorder = mimeType ? new MediaRecorder(state.audioStream, { mimeType: mimeType }) : new MediaRecorder(state.audioStream);
      state.mediaRecorder.ondataavailable = onRecorderData;
      state.mediaRecorder.onstop = function() {
        discardSpeechSegment('');
      };
      state.mediaRecorder.start(250);
      setLive(true);
      setStatus('Listening. Speak when ready.');
    } catch (err) {
      setLive(false);
      setProcessing(false);
      stopVadLoop();
      setStatus('Microphone error: ' + err.message);
    }
  }

  function stopLiveMic() {
    state.turnId += 1;
    state.phraseQueue = [];
    state.phraseText = '';
    state.llmFinished = false;
    if (state.activeRequest) {
      state.activeRequest.abort();
      state.activeRequest = null;
    }
    setLive(false);
    discardSpeechSegment('');
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
      try { state.mediaRecorder.stop(); } catch (err) {}
    }
    if (state.audioStream) {
      state.audioStream.getTracks().forEach(function(track) { track.stop(); });
      state.audioStream = null;
    }
    stopVadLoop();
    setStatus('Live mic stopped.');
  }

  function toggleLiveMic() {
    if (state.isLive) stopLiveMic();
    else startLiveMic();
  }

  function sendToLLM(audioBlob) {
    state.turnId += 1;
    state.phraseText = '';
    state.phraseQueue = [];
    state.phrasePumping = false;
    state.llmFinished = false;
    state.playbackAt = 0;
    state.pcmChunks = [];
    state.ttsStartedAt = 0;
    if (state.phraseIdleTimer) { clearTimeout(state.phraseIdleTimer); state.phraseIdleTimer = null; }
    state.ttfMs = null;
    state.ttsMs = null;
    refs.ttf.textContent = '--';
    refs.tts.textContent = '--';
    var reader = new FileReader();
    reader.onload = function() {
      var audioDataUri = reader.result;
      var userMsg = addTranscript('user', '[Voice message]');
      var systemPrompt = (refs['system-prompt'].value || '').trim();
      var startedAt = performance.now();
      fetch(config.proxyBaseUrl + '/llamacpp-audio-turn/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          endpoint: refs.endpoint.value,
          model: refs.model.value,
          api_key: refs['api-key'].value || '',
          system_prompt: systemPrompt,
          history: state.historyMessages.slice(-8),
          audio_data_url: audioDataUri,
          prompt: 'Respond to the user spoken message.',
          stream: true,
          max_tokens: 512
        })
      }).then(function(response) {
        if (!response.ok) {
          return response.text().then(function(text) {
            throw new Error(response.status + ': ' + text);
          });
        }
        var assistantMsg = addTranscript('assistant', '');
        var content = '';
        var firstToken = true;
        var streamReader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';
        var streamError = null;

        function readStream() {
          streamReader.read().then(function(result) {
            if (result.done) {
              if (streamError) throw streamError;
              if (content.trim()) {
                if (state.ttfMs !== null) {
                  assistantMsg.timing = 'LLM TTFB: ' + state.ttfMs + 'ms';
                }
                assistantMsg.content = content;
                state.historyMessages.push({ role: 'user', audio_data_url: audioDataUri, content: '' });
                state.historyMessages.push({ role: 'assistant', content: content });
                renderTranscript();
                queueProgressiveSpeech('', true);
              } else {
                setProcessing(false);
                setStatus(state.isLive ? 'The LLM returned an empty response. Listening.' : 'The LLM returned an empty response.');
              }
              return;
            }
            buffer += decoder.decode(result.value, { stream: true });
            var lines = buffer.split('\n');
            buffer = lines.pop() || '';
            lines.forEach(function(line) {
              if (line.indexOf('data: ') !== 0) return;
              var data = line.slice(6).trim();
              if (!data || data === '[DONE]') return;
              try {
                var json = JSON.parse(data);
                if (json.error) {
                  streamError = new Error(json.error.message || JSON.stringify(json.error));
                  return;
                }
                var choice = json.choices && json.choices[0];
                var delta = choice && choice.delta ? choice.delta.content || '' : '';
                if (delta) {
                  if (firstToken) {
                    state.ttfMs = Math.round(performance.now() - startedAt);
                    refs.ttf.textContent = state.ttfMs + 'ms';
                    firstToken = false;
                  }
                  content += delta;
                  queueProgressiveSpeech(delta, false);
                  assistantMsg.content = content;
                  renderTranscript();
                }
              } catch (err) {
                streamError = err;
              }
            });
            if (streamError) throw streamError;
            readStream();
          }).catch(function(err) {
            setProcessing(false);
            addTranscript('assistant', 'Error: ' + err.message);
            setStatus(state.isLive ? 'LLM stream error: ' + err.message + ' Listening.' : 'LLM stream error: ' + err.message);
          });
        }
        readStream();
      }).catch(function(err) {
        setProcessing(false);
        addTranscript('assistant', 'Error: ' + err.message);
        setStatus(state.isLive ? 'LLM error: ' + err.message + ' Listening.' : 'LLM error: ' + err.message);
      });
    };
    reader.readAsDataURL(audioBlob);
  }

  function playPcm(pcmData, sampleRate) {
    var float32Data = new Float32Array(Math.floor(pcmData.length / 2));
    for (var i = 0; i < float32Data.length; i += 1) {
      var sample = pcmData[i * 2] | (pcmData[i * 2 + 1] << 8);
      if (sample >= 32768) sample -= 65536;
      float32Data[i] = sample / 32768.0;
    }
    var audioBuffer = getAudioContext().createBuffer(1, float32Data.length, sampleRate);
    audioBuffer.getChannelData(0).set(float32Data);
    var source = getAudioContext().createBufferSource();
    source.buffer = audioBuffer;
    source.connect(getAudioContext().destination);
    var startAt = Math.max(getAudioContext().currentTime + 0.025, state.playbackAt || 0);
    source.start(startAt);
    state.playbackAt = startAt + audioBuffer.duration;
    return audioBuffer.duration;
  }

  function pcmChunksToWav(chunks, sampleRate) {
    var pcmLength = chunks.reduce(function(total, chunk) { return total + chunk.length; }, 0);
    var wav = new Uint8Array(44 + pcmLength);
    var view = new DataView(wav.buffer);
    function writeString(offset, text) {
      for (var i = 0; i < text.length; i += 1) wav[offset + i] = text.charCodeAt(i);
    }
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + pcmLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, pcmLength, true);
    var offset = 44;
    chunks.forEach(function(chunk) {
      wav.set(chunk, offset);
      offset += chunk.length;
    });
    return new Blob([wav], { type: 'audio/wav' });
  }

  function queueProgressiveSpeech(delta, final) {
    if (delta) state.phraseText += delta;
    var phraseSettings = getPhraseSettings();
    while (state.phraseText) {
      var boundary = /[.!?;:](?:\s|$)/.exec(state.phraseText);
      var cut = boundary && boundary.index + boundary[0].length >= phraseSettings.minimum
        ? boundary.index + boundary[0].length
        : (state.phraseText.length >= phraseSettings.maximum ? state.phraseText.lastIndexOf(' ', phraseSettings.maximum) : -1);
      if (cut <= 0) break;
      state.phraseQueue.push(state.phraseText.slice(0, cut).trim());
      state.phraseText = state.phraseText.slice(cut);
    }
    if (final) {
      if (state.phraseIdleTimer) { clearTimeout(state.phraseIdleTimer); state.phraseIdleTimer = null; }
      if (state.phraseText.trim()) state.phraseQueue.push(state.phraseText.trim());
      state.phraseText = '';
      state.llmFinished = true;
    } else if (state.phraseText.length >= phraseSettings.minimum) {
      if (state.phraseIdleTimer) clearTimeout(state.phraseIdleTimer);
      state.phraseIdleTimer = setTimeout(function() {
        state.phraseIdleTimer = null;
        if (state.phraseText.trim()) {
          state.phraseQueue.push(state.phraseText.trim());
          state.phraseText = '';
          pumpProgressiveSpeech();
        }
      }, phraseSettings.idleMs);
    }
    pumpProgressiveSpeech();
  }

  function pumpProgressiveSpeech() {
    if (state.phrasePumping) return;
    var phrase = state.phraseQueue.shift();
    if (!phrase) {
      if (state.llmFinished) {
        state.ttsMs = Math.round(performance.now() - state.ttsStartedAt);
        refs.tts.textContent = state.ttsMs + 'ms progressive PCM';
        if (state.pcmChunks.length) {
          if (state.aiAudioUrl) URL.revokeObjectURL(state.aiAudioUrl);
          state.aiAudioUrl = URL.createObjectURL(pcmChunksToWav(state.pcmChunks, config.sampleRate));
          showAudio(refs['ai-audio'], refs['ai-empty'], state.aiAudioUrl);
        }
        setSpeaking(false); setProcessing(false);
        setStatus(state.isLive ? 'Progressive buffered PCM playback scheduled. Listening resumes after playback.' : 'Progressive buffered PCM complete.');
      }
      return;
    }
    state.phrasePumping = true;
    setSpeaking(true);
    if (!state.ttsStartedAt) state.ttsStartedAt = performance.now();
    state.activeRequest = new AbortController();
    var currentTurn = state.turnId;
    fetch(config.proxyBaseUrl + '/audio/speech', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: state.activeRequest.signal,
      body: JSON.stringify({ input: phrase, voice: config.ttsVoice, language: config.language || 'Auto', response_format: 'pcm', stream: false })
    }).then(function(response) {
      if (!response.ok) return response.text().then(function(body) { throw new Error(response.status + ': ' + body); });
      return response.arrayBuffer();
    }).then(function(buffer) {
      if (currentTurn !== state.turnId) return;
      var pcm = new Uint8Array(buffer);
      state.pcmChunks.push(pcm);
      playPcm(pcm, config.sampleRate);
    }).catch(function(err) {
      if (err.name !== 'AbortError') setStatus('Progressive PCM error: ' + err.message);
    }).finally(function() {
      if (currentTurn === state.turnId) { state.phrasePumping = false; state.activeRequest = null; pumpProgressiveSpeech(); }
    });
  }

  function synthesizeSpeech(text) {
    setSpeaking(true);
    setStatus('Generating buffered audio.cpp TTS. Playback begins after the complete WAV is ready.');
    var ttsStartedAt = performance.now();
    state.activeRequest = new AbortController();
    fetch(config.proxyBaseUrl + '/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: state.activeRequest.signal,
      body: JSON.stringify({
        input: text,
        voice: config.ttsVoice,
        language: config.language || 'Auto',
        response_format: 'wav',
        stream: false
      })
    }).then(function(response) {
      if (!response.ok) {
        return response.text().then(function(body) {
          throw new Error(response.status + ': ' + body);
        });
      }
      return response.blob();
    }).then(function(wavBlob) {
      if (!wavBlob) return;
      state.ttsMs = Math.round(performance.now() - ttsStartedAt);
      refs.tts.textContent = state.ttsMs + 'ms buffered';
      if (state.aiAudioUrl) URL.revokeObjectURL(state.aiAudioUrl);
      state.aiAudioUrl = URL.createObjectURL(wavBlob);
      showAudio(refs['ai-audio'], refs['ai-empty'], state.aiAudioUrl);
      refs['ai-audio'].play();
      state.activeRequest = null;
      setSpeaking(false);
      setProcessing(false);
      setStatus(state.isLive ? 'Buffered playback started. Listening resumes after playback.' : 'Buffered audio ready.');
    }).catch(function(err) {
      state.activeRequest = null;
      setSpeaking(false);
      setProcessing(false);
      var message = err.name === 'AbortError' ? 'Buffered TTS cancelled.' : 'Buffered TTS error: ' + err.message;
      setStatus(state.isLive ? message + ' Listening.' : message);
    });
  }

  function clearConversation() {
    state.historyMessages = [];
    state.transcript = [];
    state.ttfMs = null;
    state.ttsMs = null;
    refs.ttf.textContent = '--';
    refs.tts.textContent = '--';
    renderTranscript();
    setStatus('Conversation cleared.');
  }

  refs.endpoint.value = localGet('sv_endpoint', 'http://host.docker.internal:8818/v1/chat/completions');
  refs.model.value = localGet('sv_model', 'gemma-4-12B-it-qat-UD-Q4_K_XL.gguf');
  refs['remember-api-key'].checked = localGet('sv_remember_api_key', 'false') === 'true';
  refs['api-key'].value = refs['remember-api-key'].checked ? localGet('sv_api_key', '') : '';
  var savedSystemPrompt = localGet('sv_system_prompt', '');
  if (savedSystemPrompt === 'You are a helpful assistant. Keep responses concise.' || savedSystemPrompt === 'You are a helpful assistant.') {
    savedSystemPrompt = '';
    localSet('sv_system_prompt', '');
  }
  refs['system-prompt'].value = savedSystemPrompt;
  loadVadSettings();
  refs['voice-name'].textContent = config.profileLabel + ' -> ' + config.ttsVoice;

  function saveLlmSettings() {
    localSet('sv_endpoint', refs.endpoint.value.trim());
    localSet('sv_model', refs.model.value.trim());
    localSet('sv_system_prompt', refs['system-prompt'].value);
    localSet('sv_remember_api_key', String(refs['remember-api-key'].checked));
    if (refs['remember-api-key'].checked) localSet('sv_api_key', refs['api-key'].value);
    else localStorage.removeItem('sv_api_key');
    refs['llm-save-status'].textContent = 'Saved.';
  }
  ['speech-threshold', 'start-hold', 'silence-delay', 'min-utterance', 'max-utterance', 'pre-roll'].forEach(function(name) {
    refs[name].addEventListener('change', saveVadSettings);
  });
  ['phrase-min', 'phrase-max', 'phrase-idle'].forEach(function(name) {
    refs[name].addEventListener('change', function() { localSet('sv_' + name, refs[name].value); });
  });
  refs['mic-select'].addEventListener('change', function() { localSet('sv_mic_id', refs['mic-select'].value); });
  refs['save-llm-settings'].addEventListener('click', saveLlmSettings);
  refs['toggle-record'].addEventListener('click', toggleLiveMic);
  refs['refresh-mics'].addEventListener('click', function() { refreshMics(true); });
  refs.clear.addEventListener('click', clearConversation);

  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', function() { refreshMics(false); });
  }
  renderTranscript();
  refreshMics(false);
})();
</script>
<img alt="" class="sv-hidden" src="x" onerror="this.onerror=null;var boot=document.getElementById('__UID__-boot');if(boot){var script=document.createElement('script');script.textContent=boot.textContent;document.body.appendChild(script);}this.remove();">
</div>'''
            return tmpl.replace("__CONFIG__", config).replace("__UID__", uid)

        # ------------------------------------------------------------------
        # Callback implementations
        # ------------------------------------------------------------------

        def on_refresh_voices(base_url: str, timeout_s: float):
            voices = try_fetch_voices(base_url, float(timeout_s))
            return (
                gr.State(base_url),
                gr.State(voices),
                gr.Dropdown(choices=voices, value=voices[0] if voices else None),
                f"✅ Loaded {len(voices)} voices from server (or fallback list).",
            )

        def fetch_backend_status(base_url: str):
            """Fetch /health for quick model verification."""
            try:
                url = f"{base_url.rstrip('/')}/health"
                r = httpx.get(url, timeout=10.0)
                r.raise_for_status()
                data = r.json()
                backend = data.get("backend") or {}
                name = backend.get("name", "unknown")
                model_id = backend.get("model_id", "unknown")
                current_key = backend.get("current_model_key")
                loaded_models = backend.get("loaded_models") or []
                runtime = backend.get("runtime") or {}
                state = runtime.get("state")
                parts = [f"**Loaded backend:** `{name}`", f"**Model:** `{model_id}`"]
                if current_key:
                    parts.append(f"**Current key:** `{current_key}`")
                if loaded_models:
                    parts.append(f"**In memory:** `{', '.join(loaded_models)}`")
                if state:
                    parts.append(f"**State:** `{state}`")
                return " | ".join(parts)
            except Exception as e:
                return f"⚠️ Could not fetch `/health`: `{e}`"

        def load_backend_models(base_url: str):
            """Fetch backend model list; show column and dropdown; display loading/error state; enable/disable Generate buttons."""
            try:
                run_id = f"ui-models-{int(datetime.utcnow().timestamp() * 1000)}"
                # region agent log
                _debug_log(run_id, "H1", "gradio_voice_studio.py:load_backend_models:entry", "load_backend_models entry", {"base_url": base_url})
                # endregion
                url = f"{base_url.rstrip('/')}/v1/backend/models"
                r = httpx.get(url, timeout=10.0)
                if r.status_code != 200:
                    # region agent log
                    _debug_log(run_id, "H2", "gradio_voice_studio.py:load_backend_models:non200", "backend models non-200", {"status_code": r.status_code, "url": url})
                    # endregion
                    return (
                        gr.update(visible=False),
                        gr.update(choices=[], value=None),
                        "Model controls are unavailable.",
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                    )
                data = r.json()
                available = [m for m in (data.get("available") or []) if "base" in m.lower()]
                if not available:
                    # region agent log
                    _debug_log(run_id, "H3", "gradio_voice_studio.py:load_backend_models:no_base", "no Base models returned", {"available_raw": data.get("available"), "state": data.get("state")})
                    # endregion
                    return (
                        gr.update(visible=False),
                        gr.update(choices=[], value=None),
                        "No Base model controls available for this backend.",
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                    )
                current = data.get("current")
                loaded = data.get("loaded_models") or []
                state = data.get("state", "unknown")
                err = data.get("last_error")
                runtime = data.get("runtime") or {}
                last_load_s = runtime.get("last_load_elapsed_s")

                # State line: loading / ready / error
                if state == "loading":
                    status_parts = ["**State:** ⏳ Loading… (wait for completion or check logs)"]
                elif state == "loaded":
                    status_parts = ["**State:** ✅ Loaded — generation enabled"]
                elif state == "error":
                    status_parts = ["**State:** ❌ Error — load a model before generating"]
                else:
                    status_parts = [f"**State:** `{state}` — load a model to enable generation"]

                status_parts.append(f"**Selected (dropdown):** `{current or 'none'}`")
                status_parts.append(f"**In memory:** `{', '.join(loaded) if loaded else 'none'}`")
                if last_load_s is not None:
                    status_parts.append(f"**Last load time:** {last_load_s}s")
                gpu = runtime.get("gpu_now") or runtime.get("gpu") or {}
                if gpu:
                    status_parts.append(
                        "**GPU:** "
                        f"{gpu.get('freeMiB', 'n/a')} MiB free / {gpu.get('requiredMiB', 'n/a')} MiB required "
                        f"({gpu.get('utilizationPercent', 'n/a')}% utilization)"
                    )
                status_parts.append(
                    f"**Memory saver:** `{'enabled' if runtime.get('mem_saver') else 'disabled'}` "
                    f"| **Synthesis reserve:** `{runtime.get('synthesis_headroom_mib', 'n/a')} MiB`"
                )
                if data.get("last_action"):
                    status_parts.append(f"**Last lifecycle action:** `{data['last_action']}`")
                events = data.get("events") or []
                if events:
                    recent = events[-5:]
                    status_parts.append("**Recent backend events:**\n" + "\n".join(
                        f"- `{event.get('at', '')}` `{event.get('action', '')}` — `{event.get('model', 'none')}`"
                        for event in recent
                    ))
                if err:
                    status_parts.append(f"**Last error:** `{err}`")

                model_ready = state == "loaded"
                # region agent log
                _debug_log(
                    run_id,
                    "H4",
                    "gradio_voice_studio.py:load_backend_models:success",
                    "backend models loaded",
                    {"state": state, "current": current, "loaded_models": loaded, "model_ready": model_ready, "last_error": err},
                )
                # endregion
                return (
                    gr.update(visible=True),
                    gr.update(choices=available, value=current or available[0]),
                    "\n\n".join(status_parts),
                    gr.update(interactive=model_ready),
                    gr.update(interactive=model_ready),
                )
            except Exception as e:
                return (
                    gr.update(visible=False),
                    gr.update(choices=[], value=None),
                    f"⚠️ Failed to load model controls: {e}",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )

        def gpu_guard_status(settings: Dict[str, Any], warning: str = "") -> str:
            mode = settings.get("mode", "enforced")
            text = (
                f"**Mode:** `{mode}` | **Load reserve:** `{settings.get('load_min_free_mib', 'n/a')} MiB` "
                f"| **Synthesis reserve:** `{settings.get('synthesis_min_free_mib', 'n/a')} MiB`\n\n"
                f"**Utilization limits:** load `{settings.get('load_max_utilization_percent', 'n/a')}%`, "
                f"synthesis `{settings.get('synthesis_max_utilization_percent', 'n/a')}%`."
            )
            if mode == "disabled":
                text += "\n\nWARNING: admission bypass is active; a CUDA out-of-memory failure remains possible."
            if warning:
                text += f"\n\n{warning}"
            return text

        def load_gpu_guard_settings(base_url: str):
            try:
                response = httpx.get(f"{base_url.rstrip('/')}/control/gpu-guard", timeout=10.0)
                response.raise_for_status()
                settings = response.json().get("gpu_guard") or {}
                return (
                    gr.update(value=settings.get("mode", "enforced")),
                    gr.update(value=settings.get("load_min_free_mib", 0)),
                    gr.update(value=settings.get("synthesis_min_free_mib", 2048)),
                    gr.update(value=settings.get("load_max_utilization_percent", 85)),
                    gr.update(value=settings.get("synthesis_max_utilization_percent", 95)),
                    gpu_guard_status(settings),
                )
            except Exception as exc:
                return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Could not load GPU guard settings: `{exc}`")

        def save_gpu_guard_settings(base_url: str, mode: str, load_reserve: float, synthesis_reserve: float, load_utilization: float, synthesis_utilization: float):
            payload = {
                "mode": mode,
                "load_min_free_mib": int(load_reserve),
                "synthesis_min_free_mib": int(synthesis_reserve),
                "load_max_utilization_percent": int(load_utilization),
                "synthesis_max_utilization_percent": int(synthesis_utilization),
            }
            try:
                response = httpx.post(f"{base_url.rstrip('/')}/control/gpu-guard", json=payload, timeout=10.0)
                if response.status_code != 200:
                    detail = response.json().get("detail", response.text)
                    raise gr.Error(str(detail))
                settings = response.json().get("gpu_guard") or payload
                return gpu_guard_status(settings, "Saved to the candidate's private persistent storage.")
            except gr.Error:
                raise
            except Exception as exc:
                raise gr.Error(f"Could not save GPU guard settings: {exc}")

        def do_switch_backend_model(base_url: str, model_key: Optional[str]):
            """POST switch model; return updated dropdown, status text, and Generate button interactive state."""
            if not model_key:
                return gr.update(), "Select a model first.", gr.update(), gr.update()
            try:
                run_id = f"ui-switch-{int(datetime.utcnow().timestamp() * 1000)}"
                # region agent log
                _debug_log(run_id, "H5", "gradio_voice_studio.py:do_switch_backend_model:entry", "switch requested", {"base_url": base_url, "model_key": model_key})
                # endregion
                url = f"{base_url.rstrip('/')}/v1/backend/models/switch"
                r = httpx.post(url, json={"model_key": model_key}, timeout=600.0)
                # region agent log
                _debug_log(run_id, "H5", "gradio_voice_studio.py:do_switch_backend_model:switch_response", "switch response received", {"status_code": r.status_code})
                # endregion
                if r.status_code != 200:
                    err = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    msg = err.get("detail", {}).get("message", "Switch failed") if isinstance(err.get("detail"), dict) else str(err.get("detail", "Switch failed"))
                    raise gr.Error(msg)
                r2 = httpx.get(f"{base_url.rstrip('/')}/v1/backend/models", timeout=10.0)
                if r2.status_code != 200:
                    return gr.update(value=model_key), f"✅ Loaded: `{model_key}`", gr.update(interactive=True), gr.update(interactive=True)
                data = r2.json()
                available = [m for m in (data.get("available", []) or []) if "base" in m.lower()]
                loaded = data.get("loaded_models") or []
                state = data.get("state", "loaded")
                model_ready = state == "loaded"
                msg = f"✅ Loaded: `{data.get('current', model_key)}` | In memory: `{', '.join(loaded) if loaded else 'none'}`"
                # region agent log
                _debug_log(run_id, "H4", "gradio_voice_studio.py:do_switch_backend_model:post_status", "post-switch models status", {"state": state, "current": data.get("current", model_key), "loaded_models": loaded})
                # endregion
                return (
                    gr.update(choices=available, value=data.get("current", model_key)),
                    msg,
                    gr.update(interactive=model_ready),
                    gr.update(interactive=model_ready),
                )
            except gr.Error:
                raise
            except Exception as e:
                raise gr.Error(str(e))

        def ensure_backend_model(base_url: str, model_key: Optional[str]) -> Optional[str]:
            """Validate that the selected model is already loaded before generation."""
            if not model_key:
                raise gr.Error("Select a model, then click Load selected model before generating.")
            status_url = f"{base_url.rstrip('/')}/v1/backend/models"
            try:
                status = httpx.get(status_url, timeout=10.0)
                if status.status_code != 200:
                    raise gr.Error(f"Could not verify loaded model: HTTP {status.status_code}")
                data = status.json()
            except gr.Error:
                raise
            except Exception as e:
                raise gr.Error(f"Could not verify loaded model: {e}")

            loaded = data.get("loaded_models") or []
            current = data.get("current")
            state = data.get("state")
            if current == model_key and state == "loaded" and model_key in loaded:
                return model_key
            if state == "unloaded" or not loaded:
                raise gr.Error(f"`{model_key}` is selected but no model is loaded. Click Load selected model first.")
            raise gr.Error(
                f"`{model_key}` is selected but `{current or 'none'}` is loaded. "
                "Click Load selected model before generating."
            )

        def on_library_table_select(table: Any, evt: gr.SelectData):
            """Copy clicked Library id cell into the Selected profile id textbox."""
            try:
                row_idx, col_idx = evt.index
                id_col = TABLE_HEADERS.index("id")
                if col_idx != id_col:
                    return gr.update()
                if evt.value:
                    return str(evt.value)
                if hasattr(table, "iloc"):
                    value = table.iloc[row_idx, id_col]
                else:
                    value = table[row_idx][id_col]
                return str(value) if value is not None else gr.update()
            except Exception:
                return gr.update()

        def do_unload_backend_model(base_url: str):
            """POST unload current backend model; return status and disable Generate buttons."""
            try:
                url = f"{base_url.rstrip('/')}/v1/backend/models/unload"
                r = httpx.post(url, timeout=60.0)
                if r.status_code != 200:
                    err = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    msg = err.get("detail", {}).get("message", "Unload failed") if isinstance(err.get("detail"), dict) else str(err.get("detail", "Unload failed"))
                    raise gr.Error(msg)
                return "✅ Model unloaded. In memory: `none` — load a model to enable generation.", gr.update(interactive=False), gr.update(interactive=False)
            except gr.Error:
                raise
            except Exception as e:
                raise gr.Error(str(e))

        def on_generate_preset(base_url: str, timeout_s: float, voice: str, language: str, instructions: str, text: str):
            payload = {
                "input": text,
                "voice": voice,
                "language": language,
                "task_type": "CustomVoice",
                "instructions": instructions or "",
                "response_format": "wav",
            }
            audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
            out_path = write_bytes_to_temp_audio(audio_bytes, ext)
            return out_path, out_path, "✅ Generated audio."

        def on_save_preset(library_dir_str: str, name: str, voice: str, language: str, instructions: str):
            if not name.strip():
                raise gr.Error("Profile name is required.")
            vp = VoiceProfile(
                profile_id=safe_profile_id(),
                name=name.strip(),
                task_type="CustomVoice",
                origin="Preset(CustomVoice)",
                created_at=now_iso(),
                language=language,
                voice=voice,
                instructions=instructions or "",
            )
            save_profile(Path(library_dir_str), vp)
            return f"✅ Saved preset profile: {vp.profile_id}"

        def on_generate_design_ref(base_url: str, timeout_s: float, language: str, instructions: str, ref_line: str):
            if not instructions.strip():
                raise gr.Error("Voice design instructions are required.")
            payload = {
                "input": ref_line,
                "voice": "Vivian",
                "language": language,
                "task_type": "VoiceDesign",
                "instructions": instructions.strip(),
                "response_format": "wav",
            }
            audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
            out_path = write_bytes_to_temp_audio(audio_bytes, ext)
            return out_path, out_path, "✅ Generated reference clip."

        def on_save_design_as_clone(
            library_dir_str: str,
            name: str,
            language: str,
            instructions: str,
            ref_line: str,
            ref_audio_path: str,
        ):
            if not name.strip():
                raise gr.Error("Profile name is required.")
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise gr.Error("Generate a reference clip first.")
            pid = safe_profile_id()
            dest_dir = profile_dir(Path(library_dir_str), pid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            ref_ext = Path(ref_audio_path).suffix or ".wav"
            ref_filename = f"ref_audio{ref_ext}"
            shutil.copy2(ref_audio_path, dest_dir / ref_filename)
            vp = VoiceProfile(
                profile_id=pid,
                name=name.strip(),
                task_type="Base",
                origin="VoiceDesign->Base",
                created_at=now_iso(),
                language=language,
                voice="Vivian",
                instructions=instructions.strip(),
                ref_text=ref_line,
                x_vector_only_mode=False,
                ref_audio_filename=ref_filename,
            )
            save_profile(Path(library_dir_str), vp)
            return f"✅ Saved designed voice as reusable clone profile: {pid}"

        def on_generate_clone(
            base_url: str,
            timeout_s: float,
            selected_model: Optional[str],
            language: str,
            ref_audio_path: str,
            ref_text: str,
            xvec_only: bool,
            text: str,
        ):
            active_model = ensure_backend_model(base_url, selected_model)
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise gr.Error("Reference audio is required.")
            if (not xvec_only) and (not ref_text.strip()):
                raise gr.Error("Reference transcript is required unless x_vector_only_mode is enabled.")
            ref_b64 = base64_from_file(Path(ref_audio_path))
            payload = {
                "input": text,
                "voice": "Vivian",
                "language": language,
                "task_type": "Base",
                "ref_audio": ref_b64,
                "ref_text": ref_text.strip(),
                "x_vector_only_mode": bool(xvec_only),
                "response_format": "wav",
            }
            # Use streaming endpoint for better timing info
            try:
                audio_bytes, ext, timing_info = request_tts_streaming(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                
                # Format timing info as readable markdown
                first_chunk = timing_info.get('first_chunk_time')
                total_time = timing_info.get('total_time')
                audio_duration = timing_info.get('audio_duration')
                rtf = timing_info.get('rtf')
                chunk_count = timing_info.get('chunk_count', 0)
                
                timing_md = f"""### ⏱️ Generation Timing
| Metric | Value |
|--------|-------|
| **First chunk** | {first_chunk:.2f}s |
| **Total time** | {total_time:.2f}s |
| **Audio duration** | {audio_duration:.2f}s |
| **RTF** | {rtf:.2f}x |
| **Chunks** | {chunk_count} |
| **Model** | {active_model or "current"} |
"""
                return out_path, out_path, timing_md
            except Exception as e:
                # Fallback to non-streaming
                audio_bytes, ext, _headers = request_tts_voice_clone(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                return out_path, out_path, f"⚠️ Used non-streaming fallback: {e}"

        def on_save_clone_profile(
            library_dir_str: str,
            name: str,
            language: str,
            ref_audio_path: str,
            ref_text: str,
            xvec_only: bool,
        ):
            if not name.strip():
                raise gr.Error("Profile name is required.")
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise gr.Error("Reference audio is required.")
            if (not xvec_only) and (not ref_text.strip()):
                raise gr.Error("Reference transcript is required unless x_vector_only_mode is enabled.")
            pid = safe_profile_id()
            dest_dir = profile_dir(Path(library_dir_str), pid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            ref_ext = Path(ref_audio_path).suffix or ".wav"
            ref_filename = f"ref_audio{ref_ext}"
            shutil.copy2(ref_audio_path, dest_dir / ref_filename)
            vp = VoiceProfile(
                profile_id=pid,
                name=name.strip(),
                task_type="Base",
                origin="Clone(Base)",
                created_at=now_iso(),
                language=language,
                voice="Vivian",
                instructions="",
                ref_text=ref_text.strip(),
                x_vector_only_mode=bool(xvec_only),
                ref_audio_filename=ref_filename,
            )
            save_profile(Path(library_dir_str), vp)
            return f"✅ Saved clone profile: {pid}"

        def profile_dropdown_update(choices: List[str], selected: Optional[str] = None):
            """Return a persistent-library dropdown update safe for Gradio validation."""
            value = selected if selected in choices else (choices[0] if choices else None)
            return gr.Dropdown(
                choices=choices,
                value=value,
                interactive=True,
                allow_custom_value=True,
            )

        def on_library_refresh(library_dir_str: str):
            profiles = list_profiles(Path(library_dir_str))
            table = profiles_table_rows(profiles)
            choices = [p.profile_id for p in profiles]
            return (
                table,
                profile_dropdown_update(choices),
                profile_dropdown_update(choices),
            )

        def on_load_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            vp = load_profile(Path(library_dir_str), pid.strip())
            ref_path = ""
            if vp.ref_audio_filename:
                candidate = profile_dir(Path(library_dir_str), vp.profile_id) / vp.ref_audio_filename
                if candidate.exists():
                    ref_path = str(candidate)
            choices = [profile.profile_id for profile in list_profiles(Path(library_dir_str))]
            return (
                vp.__dict__,
                ref_path,
                vp.name,
                vp.language,
                vp.ref_text,
                profile_dropdown_update(choices, vp.profile_id),
                profile_dropdown_update(choices, vp.profile_id),
            )

        def on_save_profile_edits(
            library_dir_str: str,
            pid: str,
            name: str,
            language: str,
            ref_text: str,
        ):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            if not name.strip():
                raise gr.Error("Profile name is required.")
            vp = load_profile(Path(library_dir_str), pid.strip())
            vp.name = name.strip()
            vp.language = language or "Auto"
            vp.ref_text = ref_text.strip()
            if vp.task_type == "Base" and not vp.x_vector_only_mode and not vp.ref_text:
                raise gr.Error("A Base profile needs a reference transcript unless x-vector-only mode is enabled.")
            save_profile(Path(library_dir_str), vp)
            return f"✅ Updated profile `{vp.profile_id}` as **{vp.name}**."

        def on_delete_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            delete_profile(Path(library_dir_str), pid.strip())
            return "✅ Deleted profile."

        def on_export_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            zip_path = export_profiles_zip(Path(library_dir_str), profile_ids=[pid.strip()])
            return zip_path, f"✅ Exported: {Path(zip_path).name}"

        def on_export_all(library_dir_str: str):
            zip_path = export_profiles_zip(Path(library_dir_str), profile_ids=None)
            return zip_path, f"✅ Exported: {Path(zip_path).name}"

        def on_import_zip(library_dir_str: str, zip_file_path: Optional[str]):
            if not zip_file_path:
                raise gr.Error("Choose a ZIP file first.")
            summary = import_profiles_zip(Path(library_dir_str), Path(zip_file_path))
            msg = f"✅ Imported {summary['imported']} profile(s); skipped {summary['skipped']}."
            if summary["errors"]:
                msg += f" Issues: {'; '.join(summary['errors'][:3])}"
                if len(summary["errors"]) > 3:
                    msg += f" (+{len(summary['errors']) - 3} more)"
            return msg

        def on_play_generate(
            base_url: str,
            timeout_s: float,
            selected_model: Optional[str],
            library_dir_str: str,
            pid: str,
            text: str,
            fmt: str,
            speed: float,
            seed: float,
        ):
            active_model = ensure_backend_model(base_url, selected_model)
            if not pid:
                raise gr.Error("Pick a saved profile.")
            vp = load_profile(Path(library_dir_str), pid)
            payload: Dict[str, Any] = {
                "input": text,
                "response_format": fmt,
                "speed": float(speed),
                "language": vp.language,
            }
            if vp.task_type == "CustomVoice":
                payload.update({
                    "task_type": "CustomVoice",
                    "voice": vp.voice,
                    "instructions": vp.instructions or "",
                })
                # Use non-streaming for CustomVoice
                audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                return out_path, out_path, ""
            elif vp.task_type == "Base":
                payload.update({
                    "task_type": "Base",
                    "voice": vp.voice or "Vivian",
                    "x_vector_only_mode": bool(vp.x_vector_only_mode),
                    "seed": int(seed) if seed is not None else -1,
                    "cache_key": vp.profile_id,
                })
                if vp.ref_audio_filename:
                    ref_file = profile_dir(Path(library_dir_str), vp.profile_id) / vp.ref_audio_filename
                    if not ref_file.exists():
                        raise gr.Error("This profile is missing its reference audio file.")
                    payload["ref_audio"] = base64_from_file(ref_file)
                else:
                    raise gr.Error("This Base profile has no stored ref_audio.")
                if not vp.x_vector_only_mode:
                    if not vp.ref_text.strip():
                        raise gr.Error("This profile needs ref_text unless x_vector_only_mode is enabled.")
                    payload["ref_text"] = vp.ref_text.strip()
                
                # Use non-streaming voice clone for full quality (avoids windowing degradation)
                try:
                    audio_bytes, ext, headers = request_tts_voice_clone(base_url, payload, float(timeout_s))
                    out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                    seed_used = headers.get("x-tts-seed") or headers.get("X-TTS-Seed")
                    seed_line = f"| **Seed used** | {seed_used} |\n" if seed_used else ""
                    format_line = (
                        f"| **Output container / codec** | {headers.get('x-tts-container', headers.get('x-tts-format', ext)).upper()} / {headers.get('x-tts-codec', 'backend')} |\n"
                        f"| **Quality path** | {headers.get('x-tts-quality', 'native backend output')} |\n"
                        f"| **Sample rate / depth** | {headers.get('x-tts-sample-rate', 'n/a')} Hz / {headers.get('x-tts-bits-per-sample', 'n/a')} bit |\n"
                        f"| **Duration** | {headers.get('x-tts-duration-seconds', 'n/a')} s |\n"
                    )
                    timing_md = f"""### ⏱️ Generation Complete (Non-streaming)
| Metric | Value |
|--------|-------|
| **Size** | {len(audio_bytes)} bytes |
| **Model** | {active_model or "current"} |
{format_line}
{seed_line}"""
                    return out_path, out_path, timing_md
                except Exception as e:
                    # This is the highest-quality path.  Do not mask a real
                    # offline synthesis error by attempting the unsupported
                    # incremental-PCM compatibility fallback.
                    raise gr.Error(f"Non-streaming generation failed: {e}")
                    # Retained legacy parser below is unreachable until a
                    # backend with native incremental PCM is introduced.
                    try:
                        audio_bytes, ext, timing_info = request_tts_streaming(base_url, payload, float(timeout_s))
                        out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                        first_chunk = timing_info.get('first_chunk_time') or 0.0
                        total_time = timing_info.get('total_time') or 0.0
                        audio_duration = timing_info.get('audio_duration') or 0.0
                        rtf = timing_info.get('rtf') or 0.0
                        chunk_count = timing_info.get('chunk_count', 0)
                        seed_used = timing_info.get('seed_used')
                        seed_line = f"| **Seed used** | {seed_used} |\n" if seed_used is not None else ""
                        timing_md = f"""### ⏱️ Generation Timing (Streaming fallback)
| Metric | Value |
|--------|-------|
| **First chunk** | {first_chunk:.2f}s |
| **Total time** | {total_time:.2f}s |
| **Audio duration** | {audio_duration:.2f}s |
| **RTF** | {rtf:.2f}x |
| **Chunks** | {chunk_count} |
| **Model** | {active_model or "current"} |
{seed_line}"""
                        return out_path, out_path, timing_md
                    except Exception as e2:
                        raise gr.Error(f"Both non-streaming and streaming failed: {e}; {e2}")
            else:
                payload.update({
                    "task_type": "VoiceDesign",
                    "voice": vp.voice or "Vivian",
                    "instructions": vp.instructions or "",
                })
                audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                return out_path, out_path, ""

        # ------------------------------------------------------------------
        # Streaming mode callbacks
        # ------------------------------------------------------------------

        def on_update_streaming_widget(base_url: str, pid: str, library_dir_str: str) -> str:
            """Render the streaming widget HTML with current TTS base URL and selected voice."""
            if not pid:
                return "<div class='svwidget'><p style='color:#999;text-align:center;padding:24px;'>Select a TTS voice profile above.</p></div>"
            try:
                vp = load_profile(Path(library_dir_str), pid)
                voice_name = vp.voice or "Vivian"
                tts_voice = f"clone:{vp.profile_id}" if vp.task_type == "Base" else voice_name
                profile_label = vp.name or vp.profile_id
                language = vp.language or "Auto"
            except Exception:
                voice_name = "Vivian"
                tts_voice = voice_name
                profile_label = pid or voice_name
                language = "Auto"
            return _build_streaming_widget_html(base_url, voice_name, tts_voice, profile_label, language)

        def on_play_mode_change(mode: str):
            """Toggle visibility of non-streaming vs streaming columns."""
            is_streaming = "Streaming" in mode
            return (
                gr.update(visible=not is_streaming),
                gr.update(visible=is_streaming),
            )

        def on_s_voice_change(pid: str, base_url: str, library_dir_str: str):
            """Update the streaming widget HTML when the voice profile changes."""
            html = on_update_streaming_widget(base_url, pid, library_dir_str)
            return gr.update(value=html)

        # ------------------------------------------------------------------
        # Wire up UI interactions to callbacks
        # ------------------------------------------------------------------

        refresh_voices_btn.click(
            fn=on_refresh_voices,
            inputs=[base_url_in, timeout_in],
            outputs=[state_base_url, state_voices, preset_voice, voices_status],
        )
        refresh_voices_btn.click(
            fn=fetch_backend_status,
            inputs=[base_url_in],
            outputs=[backend_status_md],
        )

        preset_generate_btn.click(
            fn=on_generate_preset,
            inputs=[base_url_in, timeout_in, preset_voice, preset_language, preset_instructions, preset_test_text],
            outputs=[preset_audio, preset_download, global_log],
        )
        preset_save_btn.click(
            fn=on_save_preset,
            inputs=[library_dir_in, preset_name, preset_voice, preset_language, preset_instructions],
            outputs=[global_log],
        )

        design_generate_btn.click(
            fn=on_generate_design_ref,
            inputs=[base_url_in, timeout_in, design_language, design_instructions, design_ref_line],
            outputs=[design_audio, design_download, global_log],
        )
        design_save_as_clone_btn.click(
            fn=on_save_design_as_clone,
            inputs=[library_dir_in, design_name, design_language, design_instructions, design_ref_line, design_audio],
            outputs=[global_log],
        )

        clone_generate_btn.click(
            fn=on_generate_clone,
            inputs=[base_url_in, timeout_in, backend_model_dropdown, clone_language, clone_ref_audio, clone_ref_text, clone_xvec_only, clone_test_text],
            outputs=[clone_audio, clone_download, global_log],
        )
        clone_save_btn.click(
            fn=on_save_clone_profile,
            inputs=[library_dir_in, clone_name, clone_language, clone_ref_audio, clone_ref_text, clone_xvec_only],
            outputs=[global_log],
        ).then(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id, s_voice_profile_id],
        )

        library_refresh_btn.click(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id, s_voice_profile_id],
        )
        library_table.select(
            fn=on_library_table_select,
            inputs=[library_table],
            outputs=[selected_id],
        )
        load_selected_btn.click(
            fn=on_load_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[
                profile_details,
                ref_preview,
                edit_name,
                edit_language,
                edit_ref_text,
                play_profile_id,
                s_voice_profile_id,
            ],
        )
        save_profile_edits_btn.click(
            fn=on_save_profile_edits,
            inputs=[library_dir_in, selected_id, edit_name, edit_language, edit_ref_text],
            outputs=[global_log],
        ).then(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id, s_voice_profile_id],
        )
        delete_selected_btn.click(
            fn=on_delete_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[global_log],
        ).then(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id, s_voice_profile_id],
        )

        export_selected_btn.click(
            fn=on_export_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[export_file, global_log],
        )
        export_all_btn.click(
            fn=on_export_all,
            inputs=[library_dir_in],
            outputs=[export_file, global_log],
        )

        play_generate_btn.click(
            fn=on_play_generate,
            inputs=[base_url_in, timeout_in, backend_model_dropdown, library_dir_in, play_profile_id, play_text, play_response_format, play_speed, play_seed],
            outputs=[play_audio, play_download, play_timing],
        )

        # Streaming mode wiring
        play_mode.change(
            fn=on_play_mode_change,
            inputs=[play_mode],
            outputs=[ns_group, s_group],
        )
        play_mode.change(
            fn=on_update_streaming_widget,
            inputs=[base_url_in, s_voice_profile_id, library_dir_in],
            outputs=[s_streaming_widget],
        )
        s_voice_profile_id.change(
            fn=on_s_voice_change,
            inputs=[s_voice_profile_id, base_url_in, library_dir_in],
            outputs=[s_streaming_widget],
        )

        demo.load(fn=on_library_refresh, inputs=[library_dir_in], outputs=[library_table, play_profile_id, s_voice_profile_id])
        demo.load(
            fn=load_backend_models,
            inputs=[base_url_in],
            outputs=[backend_model_column, backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        demo.load(
            fn=load_gpu_guard_settings,
            inputs=[base_url_in],
            outputs=[
                gpu_guard_mode,
                gpu_load_headroom,
                gpu_synthesis_headroom,
                gpu_load_utilization,
                gpu_synthesis_utilization,
                gpu_guard_status_md,
            ],
        )
        gpu_guard_save_btn.click(
            fn=save_gpu_guard_settings,
            inputs=[
                base_url_in,
                gpu_guard_mode,
                gpu_load_headroom,
                gpu_synthesis_headroom,
                gpu_load_utilization,
                gpu_synthesis_utilization,
            ],
            outputs=[gpu_guard_status_md],
        ).then(
            fn=load_backend_models,
            inputs=[base_url_in],
            outputs=[backend_model_column, backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        switch_model_btn.click(
            fn=do_switch_backend_model,
            inputs=[base_url_in, backend_model_dropdown],
            outputs=[backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        unload_model_btn.click(
            fn=do_unload_backend_model,
            inputs=[base_url_in],
            outputs=[backend_models_status_md, play_generate_btn, clone_generate_btn],
        ).then(
            fn=load_backend_models,
            inputs=[base_url_in],
            outputs=[backend_model_column, backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        refresh_model_status_btn.click(
            fn=load_backend_models,
            inputs=[base_url_in],
            outputs=[backend_model_column, backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        import_zip_btn.click(
            fn=on_import_zip,
            inputs=[library_dir_in, import_zip_file],
            outputs=[global_log],
        ).then(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id, s_voice_profile_id],
        )

    return demo


def main():
    """Launch the Voice Studio as a standalone web app for testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_TTS_BASE_URL)
    parser.add_argument("--library-dir", default=str(DEFAULT_LIBRARY_DIR))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app = build_app(args.base_url, Path(args.library_dir))
    app.queue(default_concurrency_limit=4).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
