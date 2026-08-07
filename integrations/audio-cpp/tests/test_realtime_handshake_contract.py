"""Contracts for the standalone copy of the strict HF Realtime handshake."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[3]
WEB = ROOT / "web" / "hf-realtime-voice"
MAIN = (WEB / "main.js").read_text(encoding="utf-8")
CLIENT = (WEB / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")
SERVER = (WEB / "server.py").read_text(encoding="utf-8")


def _provider_normalizer():
    tree = ast.parse(SERVER)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_canonical_tts_provider", "_normalize_tts_provider_settings"}
    ]
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(WEB / "server.py"), "exec"), namespace)
    return namespace["_normalize_tts_provider_settings"]


def test_candidate_rest_scope_never_enters_websocket_tuning() -> None:
    rest = MAIN.split("function candidateRestTuningPayload", 1)[1].split(
        "function activeTtsTuning", 1
    )[0]
    session = MAIN.split("function activeTtsTuning", 1)[1].split(
        "function updateRealtimeAudioSummary", 1
    )[0]

    literal = rest.split("const payload = {", 1)[1].split("};", 1)[0]
    keys = []
    for line in literal.splitlines():
        field = line.strip().rstrip(",")
        if not field:
            continue
        keys.append(field.split(":", 1)[0])
    assert keys == ["provider", "scope", "profile_id", "overrides"]
    assert 'provider: AUDIO_CPP_PROVIDER' in literal
    assert 'scope: "realtime"' in literal
    assert "payload.resolved" in rest
    assert "const { scope: _restScope, ...sessionTuning } = payload" in session
    assert "return sessionTuning" in session
    assert 'scope: "realtime"' not in session


def test_initial_configuration_ack_is_a_fail_closed_microphone_gate() -> None:
    assert "PIPELINE_CONFIG_ACK_TIMEOUT_MS = 15_000" in CLIENT
    assert "Promise.all([audioReady, wsReady, configReady])" in CLIENT
    assert 'case "pipeline.config.updated"' in CLIENT
    assert "this._resolveInitialConfig()" in CLIENT
    mic = CLIENT.split("_onMicChunk(pcm16Buffer)", 1)[1].split("_onWsMessage", 1)[0]
    assert "if (!this._sessionConfigured) return" in mic

    pre_ack_error = CLIENT.split('case "error":', 1)[1].split(
        'if (err?.type === "conversation_already_has_active_response"', 1
    )[0]
    assert "if (!this._sessionConfigured)" in pre_ack_error
    assert "this._rejectInitialConfig(failure)" in pre_ack_error
    assert "await this.close()" in pre_ack_error


def test_candidate_preflight_requires_live_profile_resolution() -> None:
    preflight = MAIN.split("async function assertTtsBackendReady", 1)[1].split(
        "async function assertModelEndpointReady", 1
    )[0]
    assert "await refreshCandidateTuningProfiles();" in preflight
    assert ".catch(" not in preflight


def test_provider_switch_invalidates_stale_profile_and_voice_state_before_await() -> None:
    change = MAIN.split('inputTtsBackend.addEventListener("change"', 1)[1].split(
        "await Promise.all", 1
    )[0]
    assert "settings.ttsBackend = normalizeTtsProvider(inputTtsBackend.value)" in change
    assert "candidateTuningResolved = null" in change
    assert "voiceInventoryRequest += 1" in change
    assert "clearVoiceProfileOptions(settings.ttsBackend" in change

    active = MAIN.split("function activeTtsTuning", 1)[1].split(
        "function updateRealtimeAudioSummary", 1
    )[0]
    assert "normalizeTtsProvider(backend) !== AUDIO_CPP_PROVIDER" in active
    assert "return null" in active


def test_legacy_audio_cpp_provider_is_canonicalized_before_persistence() -> None:
    normalize = _provider_normalizer()
    saved = normalize(
        {
            "ttsBackend": "audio-cpp",
            "voiceByBackend": {"audio-cpp": "clone:legacy-profile"},
            "ttsProfileByBackend": {"audio-cpp": "balanced"},
        }
    )

    assert saved["ttsBackend"] == "qwen3tts-audiocpp"
    assert saved["voiceByBackend"] == {"qwen3tts-audiocpp": "clone:legacy-profile"}
    assert saved["ttsProfileByBackend"] == {"qwen3tts-audiocpp": "balanced"}
    writer = SERVER.split("def _write_public_ui_settings", 1)[1].split(
        "def _read_audio_cpp_validation", 1
    )[0]
    assert "payload = _normalize_tts_provider_settings(payload)" in writer
