"""Behavioral contracts for paired clone references and offline policy isolation."""
from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import wave
import unittest


SUPERVISOR = Path(__file__).parents[1] / "supervisor.py"
SPEC = importlib.util.spec_from_file_location("candidate_supervisor_reference_test", SUPERVISOR)
assert SPEC and SPEC.loader
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def wav_bytes(seconds: int, *, rate: int = 10) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x00\x00" * seconds * rate)
    return output.getvalue()


class _FakeResponse:
    def __init__(
        self, content: bytes, chunks: list[bytes] | None = None,
        decode_mode: str | None = None,
    ) -> None:
        self.content = content
        self.headers = {"content-type": "audio/wav"}
        if decode_mode:
            self.headers[supervisor.ENGINE_DECODE_MODE_HEADER] = decode_mode
        self.status_code = 200
        self.is_error = False
        self._chunks = chunks or [b"\x01\x00", b"\x02\x00"]

    async def aread(self) -> bytes:
        return self.content

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _FakeAsyncClient:
    requests: list[dict] = []
    references: list[tuple[str, float, str]] = []
    output = wav_bytes(2, rate=24000)

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    @classmethod
    def _capture(cls, payload: dict) -> None:
        reference = Path(payload["voice_ref"])
        with wave.open(str(reference), "rb") as reader:
            duration = reader.getnframes() / reader.getframerate()
        cls.references.append((str(reference), duration, str(payload["reference_text"])))
        cls.requests.append(copy.deepcopy(payload))

    async def post(self, _url: str, json: dict) -> _FakeResponse:
        self._capture(json)
        decode_mode = None
        if json.get("options", {}).get("qwen3_tts.decode_mode") == "offline_full":
            decode_mode = supervisor.OFFLINE_FULL_DECODE_MODE
        return _FakeResponse(self.output, decode_mode=decode_mode)

    def stream(self, _method: str, _url: str, json: dict) -> _FakeResponse:
        self._capture(json)
        return _FakeResponse(self.output, decode_mode="native-incremental-pcm")


class ReferencePairingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_client = supervisor.httpx.AsyncClient
        self.original_read = supervisor._read_tuning_profiles
        self.original_native = supervisor.NATIVE_INCREMENTAL_PCM_ENABLED
        self.document = supervisor._default_tuning_profiles()
        supervisor.httpx.AsyncClient = _FakeAsyncClient
        supervisor._read_tuning_profiles = lambda: self.document
        supervisor.NATIVE_INCREMENTAL_PCM_ENABLED = True
        _FakeAsyncClient.requests.clear()
        _FakeAsyncClient.references.clear()
        self.full_reference = base64.b64encode(wav_bytes(18)).decode("ascii")

    def tearDown(self) -> None:
        supervisor.httpx.AsyncClient = self.original_client
        supervisor._read_tuning_profiles = self.original_read
        supervisor.NATIVE_INCREMENTAL_PCM_ENABLED = self.original_native

    def payload(self, *, response_format: str, stream: bool) -> dict:
        payload = {
            "input": "Requested assistant text only.",
            "ref_audio": self.full_reference,
            "ref_text": "Complete eighteen second reference transcript.",
            "response_format": response_format,
            "stream": stream,
            "tuning": {
                "provider": supervisor.TUNING_PROVIDER,
                "profile_id": "low-latency",
                "overrides": {"max_reference_seconds": 12},
            },
        }
        payload["_tuning_snapshot"] = supervisor._resolve_request_tuning(payload)
        return payload

    def assert_full_pair_headers(self, response, delivery: str) -> None:
        headers = response.headers
        self.assertEqual(headers["x-tts-reference-source-seconds"], "18.0")
        self.assertEqual(headers["x-tts-reference-requested-limit-seconds"], "12")
        self.assertEqual(headers["x-tts-reference-used-seconds"], "18.0")
        self.assertEqual(headers["x-tts-reference-limit-applied"], "false")
        self.assertEqual(headers["x-tts-reference-pairing"], "full")
        self.assertEqual(headers["x-tts-reference-truncated"], "false")
        self.assertEqual(headers["x-tts-delivery-mode"], delivery)

    def assert_last_reference_cleaned(self) -> None:
        path, duration, transcript = _FakeAsyncClient.references[-1]
        self.assertEqual(duration, 18.0)
        self.assertEqual(transcript, "Complete eighteen second reference transcript.")
        self.assertFalse(Path(path).exists())

    def test_18_second_pair_is_not_audio_only_cropped_in_full_wav_or_buffered_pcm(self) -> None:
        full = asyncio.run(supervisor._voice_clone_response(self.payload(response_format="wav", stream=False)))
        self.assert_full_pair_headers(full, "offline-full-decoder")
        self.assert_last_reference_cleaned()

        buffered = asyncio.run(supervisor._voice_clone_response(self.payload(response_format="pcm", stream=False)))
        self.assert_full_pair_headers(buffered, "buffered-fallback")
        self.assert_last_reference_cleaned()

    def test_18_second_pair_is_not_audio_only_cropped_in_native_pcm(self) -> None:
        response = supervisor._native_clone_pcm_response(
            self.payload(response_format="pcm", stream=True),
            "qwen3-tts-1.7b-base-bf16",
        )
        self.assert_full_pair_headers(response, "native-incremental-pcm")

        async def consume() -> bytes:
            return b"".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(asyncio.run(consume()), b"\x01\x00\x02\x00")
        self.assert_last_reference_cleaned()

    def test_matched_excerpt_can_apply_to_streaming_but_never_full_wav(self) -> None:
        excerpt = {
            "ref_audio": base64.b64encode(wav_bytes(12)).decode("ascii"),
            "ref_text": "Exact twelve second excerpt transcript.",
        }
        buffered_payload = self.payload(response_format="pcm", stream=False)
        buffered_payload["_matched_reference_pairs"] = [excerpt]
        buffered = asyncio.run(supervisor._voice_clone_response(buffered_payload))
        self.assertEqual(buffered.headers["x-tts-reference-pairing"], "matched-excerpt")
        self.assertEqual(buffered.headers["x-tts-reference-used-seconds"], "12.0")
        self.assertEqual(buffered.headers["x-tts-reference-limit-applied"], "true")
        self.assertEqual(_FakeAsyncClient.references[-1][2], excerpt["ref_text"])

        full_payload = self.payload(response_format="wav", stream=False)
        full_payload["_matched_reference_pairs"] = [excerpt]
        full = asyncio.run(supervisor._voice_clone_response(full_payload))
        self.assert_full_pair_headers(full, "offline-full-decoder")
        self.assert_last_reference_cleaned()

    def test_full_wav_uses_fixed_quality_sampler_and_no_streaming_fields(self) -> None:
        snapshot = self.payload(response_format="wav", stream=False)["_tuning_snapshot"]
        quality = supervisor._default_tuning_profiles()["profiles"]["quality"]
        self.assertEqual(snapshot["policy"], "offline-full-quality")
        self.assertEqual(snapshot["delivery_mode"], "offline-full-decoder")
        self.assertEqual(
            snapshot["engine_fields"],
            {key: quality[key] for key in ("temperature", "top_k", "top_p", "repetition_penalty")},
        )
        for field in ("max_reference_seconds", "first_block_frames", "steady_block_frames", "left_context_frames", "text_lookahead", "phrase_flush_ms"):
            self.assertNotIn(field, snapshot["effective"])
        self.assertEqual(snapshot["phrase_queue_fields"], {})

    def test_profile_metadata_stores_and_reloads_only_explicit_matched_pairs(self) -> None:
        original_library = supervisor.VOICE_LIBRARY_DIR
        with tempfile.TemporaryDirectory() as directory:
            supervisor.VOICE_LIBRARY_DIR = Path(directory)
            try:
                supervisor._write_candidate_profile(
                    "paired",
                    {
                        "name": "Paired",
                        "ref_audio": self.full_reference,
                        "ref_text": "Complete eighteen second reference transcript.",
                        "reference_excerpts": [
                            {
                                "ref_audio": base64.b64encode(wav_bytes(12)).decode("ascii"),
                                "ref_text": "Exact twelve second excerpt transcript.",
                            }
                        ],
                    },
                )
                metadata = (Path(directory) / "profiles" / "paired" / "meta.json").read_text(encoding="utf-8")
                self.assertIn('"ref_audio_filename": "ref_excerpt_1.wav"', metadata)
                self.assertIn("Exact twelve second excerpt transcript.", metadata)
                payload = {"voice": "clone:paired"}
                supervisor._apply_clone_profile(payload)
                self.assertEqual(len(payload["_matched_reference_pairs"]), 1)
                self.assertEqual(
                    payload["_matched_reference_pairs"][0]["ref_text"],
                    "Exact twelve second excerpt transcript.",
                )
            finally:
                supervisor.VOICE_LIBRARY_DIR = original_library

    def test_builtins_migrate_to_72_frames_and_reduced_custom_context_warns(self) -> None:
        saved = supervisor._default_tuning_profiles()
        for profile in saved["profiles"].values():
            profile["left_context_frames"] = 25
            profile["revision"] = 1
        normalized = supervisor._normalize_tuning_document(saved)
        self.assertEqual(
            {profile["left_context_frames"] for profile in normalized["profiles"].values()},
            {supervisor.MODEL_REQUIRED_LEFT_CONTEXT_FRAMES},
        )

        payload = self.payload(response_format="pcm", stream=True)
        payload["tuning"]["overrides"]["left_context_frames"] = 25
        snapshot = supervisor._resolve_request_tuning(payload)
        self.assertTrue(any("Reduced decoder context" in warning for warning in snapshot["warnings"]))


if __name__ == "__main__":
    unittest.main()
