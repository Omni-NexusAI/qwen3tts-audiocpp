"""Behavioral checks for opt-in native PCM wiring and rollback boundaries."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest

from fastapi import HTTPException


ROOT = Path(__file__).parents[3]
INTEGRATION = Path(__file__).parents[1]
UI = INTEGRATION / "gradio_voice_studio.py"
SUPERVISOR = INTEGRATION / "supervisor.py"
SERVER = ROOT / "web" / "hf-realtime-voice" / "server.py"

gradio_stub = types.ModuleType("gradio")
gradio_stub.Blocks = object
gradio_stub.themes = types.SimpleNamespace(Base=object)
sys.modules.setdefault("gradio", gradio_stub)

ui_spec = importlib.util.spec_from_file_location("candidate_gradio_native_test", UI)
assert ui_spec and ui_spec.loader
studio = importlib.util.module_from_spec(ui_spec)
ui_spec.loader.exec_module(studio)

supervisor_spec = importlib.util.spec_from_file_location("candidate_supervisor_native_test", SUPERVISOR)
assert supervisor_spec and supervisor_spec.loader
supervisor = importlib.util.module_from_spec(supervisor_spec)
supervisor_spec.loader.exec_module(supervisor)


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.headers = {
            "x-tts-sample-rate": "24000",
            "x-tts-streaming-mode": "native-incremental-pcm",
        }
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.exited = True

    def raise_for_status(self) -> None:
        return None

    def iter_raw(self):
        yield from self.chunks


class _FakeClient:
    response: _FakeResponse
    request: tuple[str, str, dict] | None = None

    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def stream(self, method: str, url: str, json: dict):
        type(self).request = (method, url, json)
        return type(self).response


class _CancelAfterFirstChunk:
    def __init__(self) -> None:
        self.cancelled = False

    def is_set(self) -> bool:
        return self.cancelled


class CandidateNativeWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_client = studio.httpx.Client
        self.original_native = studio.NATIVE_INCREMENTAL_PCM_ENABLED
        self.original_supervisor_native = supervisor.NATIVE_INCREMENTAL_PCM_ENABLED
        studio.httpx.Client = _FakeClient
        studio.NATIVE_INCREMENTAL_PCM_ENABLED = True

    def tearDown(self) -> None:
        studio.httpx.Client = self.original_client
        studio.NATIVE_INCREMENTAL_PCM_ENABLED = self.original_native
        supervisor.NATIVE_INCREMENTAL_PCM_ENABLED = self.original_supervisor_native

    def test_python_stream_helper_uses_raw_native_contract_and_reports_chunks(self) -> None:
        _FakeClient.response = _FakeResponse([b"\x01\x00", b"\x02\x00"])
        observed: list[bytes] = []
        wav, extension, timing = studio.request_tts_streaming(
            "http://candidate",
            {"input": "hello", "voice": "clone:test", "response_format": "wav", "stream": False},
            30,
            on_chunk=observed.append,
        )
        self.assertEqual(_FakeClient.request[0:2], ("POST", "http://candidate/v1/audio/speech"))
        self.assertTrue(_FakeClient.request[2]["stream"])
        self.assertEqual(_FakeClient.request[2]["response_format"], "pcm")
        self.assertEqual(observed, [b"\x01\x00", b"\x02\x00"])
        self.assertEqual(extension, "wav")
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertEqual(timing["chunk_count"], 2)
        self.assertEqual(timing["pcm_bytes"], 4)
        self.assertEqual(timing["streaming_mode"], "native-incremental-pcm")

    def test_python_stream_cancellation_closes_response_before_late_pcm(self) -> None:
        _FakeClient.response = _FakeResponse([b"\x01\x00", b"\x02\x00"])
        cancel = _CancelAfterFirstChunk()
        observed: list[bytes] = []

        def receive(block: bytes) -> None:
            observed.append(block)
            cancel.cancelled = True

        with self.assertRaises(studio.NativeStreamingCancelled):
            studio.request_tts_streaming(
                "http://candidate", {"input": "cancel", "voice": "clone:test"}, 30,
                cancel_event=cancel, on_chunk=receive,
            )
        self.assertEqual(observed, [b"\x01\x00"])
        self.assertTrue(_FakeClient.response.exited)

    def test_seed_is_uint32_and_xvector_remains_unsupported(self) -> None:
        payload = {"seed": 7}
        supervisor._normalize_gradio_seed(payload)
        self.assertEqual(payload["seed"], 7)
        with self.assertRaises(HTTPException) as error:
            asyncio.run(supervisor._voice_clone_response({"x_vector_only_mode": True}))
        self.assertEqual(error.exception.status_code, 422)
        with self.assertRaises(HTTPException) as imported:
            supervisor._write_candidate_profile(
                "xvector-test",
                {"ref_audio": "AA==", "ref_text": "reference", "x_vector_only_mode": True},
            )
        self.assertEqual(imported.exception.status_code, 422)

    def test_browser_proxy_and_widget_preserve_streaming_and_cancellation(self) -> None:
        server = SERVER.read_text(encoding="utf-8")
        ui = UI.read_text(encoding="utf-8")
        self.assertIn("upstream = await http.send(upstream_request, stream=True)", server)
        self.assertIn("async for chunk in upstream.aiter_raw()", server)
        self.assertIn("await upstream.aclose()", server)
        self.assertIn("response.body.getReader()", ui)
        self.assertIn("stream: !!config.nativeStreaming", ui)
        self.assertIn("tuning: config.tuning", ui)
        self.assertIn('session_tuning.get("profile_id") == tuning_profile_id', ui)
        self.assertIn("reader.cancel()", ui)
        self.assertIn("cancelScheduledPlayback()", ui)
        self.assertIn("activeLlmRequest", ui)
        self.assertIn("activeTtsRequest", ui)
        self.assertIn("Native incremental PCM (experimental)", ui)
        self.assertIn("Buffered phrase PCM (rollback fallback)", ui)

    def test_native_enablement_is_candidate_only_and_default_is_buffered(self) -> None:
        candidate = (INTEGRATION / "compose.candidate.yml").read_text(encoding="utf-8")
        native = (INTEGRATION / "compose.native.yml").read_text(encoding="utf-8")
        self.assertIn('AUDIO_CPP_NATIVE_INCREMENTAL_PCM: "false"', candidate)
        self.assertIn('AUDIO_CPP_NATIVE_INCREMENTAL_PCM: "true"', native)
        self.assertIn('AUDIO_CPP_NATIVE_LOAD_WARMUP: "true"', native)
        self.assertIn('AUDIO_CPP_NATIVE_LOAD_WARMUP_TIMEOUT_SECONDS: "180"', native)
        self.assertIn('GGML_CUDA_DISABLE_GRAPHS: "1"', native)
        self.assertIn("${AUDIO_CPP_IMAGE:-local/audio-cpp-qwen3-tts-voice-studio:native-development}", candidate)
        self.assertNotIn("image:", native)
        self.assertIn("  audio-cpp-candidate:", native)
        self.assertNotIn("  qwen3-tts-faster:", native)
        self.assertNotIn("  groxaxo:", native.lower())

    def test_native_load_warmup_consumes_private_pcm_before_ready(self) -> None:
        originals = {
            "gpu_guard": supervisor.gpu_guard,
            "inventory": supervisor._candidate_profile_response,
            "profile": supervisor._candidate_profile_payload,
            "resolve": supervisor._resolve_request_tuning,
            "native": supervisor._native_clone_pcm_response,
        }
        captured: dict[str, object] = {}

        async def blocks():
            yield b"\x01\x00"
            yield b"\x02\x00"

        try:
            supervisor.gpu_guard = lambda *_args, **_kwargs: {"ok": True}
            supervisor._candidate_profile_response = lambda: {
                "selectedVoice": "",
                "defaultVoice": "clone:warm-profile",
                "voices": [{"id": "warm-profile"}],
            }
            supervisor._candidate_profile_payload = lambda profile_id, include_audio: {
                "id": profile_id,
                "ref_audio": "UklGRg==",
                "ref_text": "paired transcript",
            }
            supervisor._resolve_request_tuning = lambda payload: {"resolved": True}

            def fake_native(payload, model_id):
                captured.update(payload=payload, model_id=model_id)
                return types.SimpleNamespace(body_iterator=blocks())

            supervisor._native_clone_pcm_response = fake_native
            result = asyncio.run(supervisor._run_native_load_warmup("qwen3-tts-1.7b-base-bf16"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["profileId"], "warm-profile")
            self.assertEqual(captured["model_id"], "qwen3-tts-1.7b-base-bf16")
            payload = captured["payload"]
            self.assertEqual(payload["input"], "Ready.")
            self.assertEqual(payload["tuning"]["overrides"]["seed"], 321)
            self.assertEqual(payload["_tuning_snapshot"], {"resolved": True})
            self.assertTrue(payload["_internal_warmup"])
            self.assertEqual(payload["_engine_epoch"], supervisor.state["engineEpoch"])
        finally:
            supervisor.gpu_guard = originals["gpu_guard"]
            supervisor._candidate_profile_response = originals["inventory"]
            supervisor._candidate_profile_payload = originals["profile"]
            supervisor._resolve_request_tuning = originals["resolve"]
            supervisor._native_clone_pcm_response = originals["native"]

    def test_generation_admission_rejects_stale_epoch_and_allows_internal_warmup(self) -> None:
        original_state = dict(supervisor.state)
        try:
            supervisor.state.update(activeModel="qwen3-tts-1.7b-base-bf16", state="loaded", engineEpoch=7)
            supervisor._revalidate_generation_admission(
                {"_engine_epoch": 7}, "qwen3-tts-1.7b-base-bf16"
            )
            with self.assertRaises(supervisor.HTTPException) as stale:
                supervisor._revalidate_generation_admission(
                    {"_engine_epoch": 6}, "qwen3-tts-1.7b-base-bf16"
                )
            self.assertEqual(stale.exception.status_code, 409)
            supervisor.state["state"] = "warming"
            supervisor._revalidate_generation_admission(
                {"_engine_epoch": 7},
                "qwen3-tts-1.7b-base-bf16",
                internal_warmup=True,
            )
            with self.assertRaises(supervisor.HTTPException):
                supervisor._revalidate_generation_admission(
                    {"_engine_epoch": 7}, "qwen3-tts-1.7b-base-bf16"
                )
        finally:
            supervisor.state.clear()
            supervisor.state.update(original_state)

    def test_thin_overlay_rebuilds_and_copies_the_patched_engine(self) -> None:
        overlay = (INTEGRATION / "Dockerfile.overlay").read_text(encoding="utf-8")
        self.assertIn("AS engine-build", overlay)
        self.assertIn("AUDIO_CPP_REF=238ab6a9e321c17de8e120559f57efeedaeb1345", overlay)
        self.assertIn("qwen3-native-pcm-streaming.patch", overlay)
        self.assertIn("--target audiocpp_server", overlay)
        self.assertIn(
            "COPY --from=engine-build /opt/audio.cpp/build/linux-cuda-release/bin/audiocpp_server /usr/local/bin/audiocpp_server",
            overlay,
        )

    def test_model_inventory_mode_matches_both_engine_configuration_states(self) -> None:
        original_models = supervisor._models
        supervisor._models = lambda: {
            "qwen3-tts-0.6b-base-bf16": {},
            "qwen3-tts-1.7b-base-bf16": {},
        }
        try:
            for enabled, expected in ((False, "offline"), (True, "streaming")):
                with self.subTest(enabled=enabled):
                    supervisor.NATIVE_INCREMENTAL_PCM_ENABLED = enabled
                    inventory = asyncio.run(supervisor.models())
                    self.assertEqual({item["mode"] for item in inventory["data"]}, {expected})
                    self.assertEqual(supervisor._configured_model_mode(), expected)
        finally:
            supervisor._models = original_models

    def test_enforced_load_reserves_residency_plus_unchanged_synthesis_floor(self) -> None:
        original_check_output = supervisor.subprocess.check_output
        original_settings = dict(supervisor.gpu_guard_settings)
        reading = {"free": 7266, "utilization": 0}
        supervisor.subprocess.check_output = lambda *_args, **_kwargs: (
            f"{reading['free']}, {reading['utilization']}"
        )
        try:
            supervisor.gpu_guard_settings.clear()
            supervisor.gpu_guard_settings.update(supervisor._default_gpu_guard_settings(), mode="enforced")
            blocked = supervisor.gpu_guard("qwen3-tts-0.6b-base-bf16", operation="load")
            self.assertFalse(blocked["ok"])
            self.assertEqual(blocked["requiredMiB"], 8048)
            self.assertEqual(blocked["residencyReserveMiB"], 6000)
            self.assertEqual(blocked["postLoadSynthesisReserveMiB"], 2048)
            self.assertIn("6000 MiB measured/rounded model/graph residency", blocked["reason"])

            reading["free"] = 8944
            admitted = supervisor.gpu_guard("qwen3-tts-0.6b-base-bf16", operation="load")
            self.assertTrue(admitted["ok"])
            self.assertEqual(supervisor._guard_policy("qwen3-tts-1.7b-base-bf16", "load")[0], 10500)
            reading["free"] = 10499
            blocked_17b = supervisor.gpu_guard("qwen3-tts-1.7b-base-bf16", operation="load")
            self.assertFalse(blocked_17b["ok"])
            self.assertEqual(blocked_17b["admissionKind"], "existing-total-threshold")
            self.assertIsNone(blocked_17b["residencyReserveMiB"])
            self.assertIn("residency delta has not been measured", blocked_17b["reason"])
            self.assertEqual(supervisor._guard_policy("qwen3-tts-0.6b-base-bf16", "synthesis")[0], 2048)

            supervisor.gpu_guard_settings.update(
                mode="custom", load_min_free_mib=7000, synthesis_min_free_mib=3000
            )
            reading["free"] = 7266
            custom = supervisor.gpu_guard("qwen3-tts-0.6b-base-bf16", operation="load")
            self.assertTrue(custom["ok"])
            self.assertEqual(custom["requiredMiB"], 7000)
            self.assertTrue(custom["customAbsoluteLoadThreshold"])
            self.assertEqual(supervisor._guard_policy("qwen3-tts-0.6b-base-bf16", "synthesis")[0], 3000)

            supervisor.gpu_guard_settings["mode"] = "disabled"
            reading["free"] = 100
            bypassed = supervisor.gpu_guard("qwen3-tts-0.6b-base-bf16", operation="load")
            self.assertTrue(bypassed["ok"])
            self.assertTrue(bypassed["bypassed"])
        finally:
            supervisor.subprocess.check_output = original_check_output
            supervisor.gpu_guard_settings.clear()
            supervisor.gpu_guard_settings.update(original_settings)

    def test_voice_studio_probe_uses_health_and_clone_adapter_contracts(self) -> None:
        server = SERVER.read_text(encoding="utf-8")
        probe = server.split('async def voice_studio_test', 1)[1].split('async def _probe_tts_backend', 1)[0]
        self.assertIn('current_model = backend.get("model_id") or backend.get("current_model_key")', probe)
        self.assertIn('current_model != req.model_id', probe)
        self.assertIn('"voice": f"clone:{req.profile_id}"', probe)
        self.assertIn('"response_format": "wav"', probe)
        self.assertNotIn('runtime.get("current")', probe)


if __name__ == "__main__":
    unittest.main()
