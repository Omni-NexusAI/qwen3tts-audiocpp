"""Headers may go out on the engine's own label of the first block, never on a guess.

A request that pins the lifecycle pair keeps the two-delta proof unchanged.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import unittest
from pathlib import Path

INTEGRATION = Path(__file__).parents[1]
SUPERVISOR = INTEGRATION / "supervisor.py"

spec = importlib.util.spec_from_file_location("candidate_supervisor_attested_block_test", SUPERVISOR)
assert spec and spec.loader
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)

FRAME = b"\x01\x00" * 1920
MODE = 'data: {"type":"speech.decode_mode","mode":"native-incremental-pcm"}'
DONE = ['data: {"type":"speech.audio.done"}', "", "data: [DONE]", ""]


def delta(count: int, **label) -> str:
    event = {"type": "speech.audio.delta", "audio": base64.b64encode(FRAME * count).decode(), **label}
    return "data: " + json.dumps(event)


def generation(frames: int) -> str:
    return "data: " + json.dumps(
        {"type": "speech.generation", "generated_frames": frames, "generation_cap": 150, "termination": "eos"})


class FakeAsyncResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.headers: dict[str, str] = {}
        self.status_code = 200
        self.is_error = False

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aclose(self) -> None:
        return None


class FakeAsyncClient:
    lines: list[str] = []

    def __init__(self, **_kwargs) -> None:
        pass

    def build_request(self, method: str, url: str, json: dict):
        return (method, url, json)

    async def send(self, _request, stream: bool = False):
        return FakeAsyncResponse(type(self).lines)

    async def aclose(self) -> None:
        return None


class AttestedBlockProofTests(unittest.TestCase):
    def setUp(self) -> None:
        self.originals = {
            "client": supervisor.httpx.AsyncClient,
            "prepare": supervisor._prepare_reference_pair,
            "revalidate": supervisor._revalidate_generation_admission,
            "record": supervisor._record_event,
        }
        self.reference = INTEGRATION / ".attested-block-test-reference.wav"
        self.events: list[tuple[str, dict]] = []
        supervisor.httpx.AsyncClient = FakeAsyncClient
        supervisor._prepare_reference_pair = lambda *_args, **_kwargs: (
            self.reference,
            {"source_seconds": 1.0, "requested_limit_seconds": None, "used_seconds": 1.0,
             "limit_applied": False, "pairing": "full", "truncated": False},
            "reference",
        )
        supervisor._revalidate_generation_admission = lambda *_args, **_kwargs: None
        supervisor._record_event = lambda name, **detail: self.events.append((name, detail))

    def tearDown(self) -> None:
        supervisor.httpx.AsyncClient = self.originals["client"]
        supervisor._prepare_reference_pair = self.originals["prepare"]
        supervisor._revalidate_generation_admission = self.originals["revalidate"]
        supervisor._record_event = self.originals["record"]
        self.reference.unlink(missing_ok=True)

    def relay(self, lines: list[str], *, pinned: bool = False):
        async def run():
            FakeAsyncClient.lines = lines
            payload = {"input": "test", "ref_audio": "placeholder", "voice": "clone:test", "_engine_epoch": 1}
            if pinned:
                payload["_lifecycle_pair_supplied"] = True
            chunks: list[bytes] = []
            try:
                response = await supervisor._native_clone_pcm_response(payload, "qwen3-tts-1.7b-base-bf16")
                headers = dict(response.headers)
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
            except BaseException as exc:
                return chunks, {}, exc
            return chunks, headers, None

        self.events.clear()
        return asyncio.run(run())

    def test_only_the_engines_own_label_of_block_zero_is_a_proof(self) -> None:
        proof = supervisor._attested_block_proof
        self.assertEqual(proof({"seq": 0, "frames": 2, "phase": "first"}, FRAME * 2), "engine-attested-first-block")
        self.assertEqual(proof({"seq": 0, "frames": 1, "phase": "final"}, FRAME), "engine-attested-single-block")
        for event, chunk in (
            ({}, FRAME),  # an older engine sends no label
            ({"seq": 1, "frames": 1, "phase": "steady"}, FRAME),
            ({"seq": 0, "frames": 1, "phase": "steady"}, FRAME),
            ({"seq": 0, "frames": 2, "phase": "first"}, FRAME),  # byte count contradicts frames
            ({"seq": 0, "frames": 0, "phase": "first"}, b""),
            ({"seq": False, "frames": 1, "phase": "first"}, FRAME),
            ({"seq": 0, "frames": True, "phase": "first"}, FRAME),
            ({"seq": "0", "frames": 1, "phase": "first"}, FRAME),
            ({"seq": 0, "frames": 1.0, "phase": "first"}, FRAME),
            ({"seq": 0, "frames": 1}, FRAME),
        ):
            self.assertIsNone(proof(event, chunk), event)

    def test_headers_go_out_after_the_first_labelled_block(self) -> None:
        lines = [MODE, delta(2, seq=0, frames=2, phase="first"), delta(8, seq=1, frames=8, phase="steady"),
                 delta(1, seq=2, frames=1, phase="final"), generation(11), *DONE]
        chunks, headers, failure = self.relay(lines)
        self.assertIsNone(failure)
        self.assertEqual(chunks, [FRAME * 2, FRAME * 8, FRAME])
        self.assertEqual(headers["x-tts-native-engine-chunk-proof"], "engine-attested-first-block")
        self.assertEqual(headers["x-tts-streaming-mode"], "native-incremental-pcm")
        self.assertEqual(headers["x-tts-native-preheader-proof-additional-wait-ms"], "0.0")
        proof_events = [detail for name, detail in self.events if name == "native-pcm-preheader-proof"]
        self.assertEqual(proof_events[0]["preheaderProofChunks"], 1)

    def test_an_utterance_shorter_than_one_block_is_accepted_as_a_single_block(self) -> None:
        chunks, headers, failure = self.relay([MODE, delta(1, seq=0, frames=1, phase="final"), generation(1), *DONE])
        self.assertIsNone(failure)
        self.assertEqual(chunks, [FRAME])
        self.assertEqual(headers["x-tts-native-engine-chunk-proof"], "engine-attested-single-block")

    def test_a_request_that_pins_the_lifecycle_pair_keeps_the_two_delta_proof(self) -> None:
        lines = [MODE, delta(2, seq=0, frames=2, phase="first"), delta(8, seq=1, frames=8, phase="steady"),
                 generation(10), *DONE]
        chunks, headers, failure = self.relay(lines, pinned=True)
        self.assertIsNone(failure)
        self.assertEqual(chunks, [FRAME * 2, FRAME * 8])
        self.assertEqual(headers["x-tts-native-engine-chunk-proof"], "two-distinct-sse-delta-events")
        proof_events = [detail for name, detail in self.events if name == "native-pcm-preheader-proof"]
        self.assertEqual(proof_events[0]["preheaderProofChunks"], 2)

        _chunks, _headers, failure = self.relay(
            [MODE, delta(1, seq=0, frames=1, phase="final"), generation(1), *DONE], pinned=True)
        self.assertIsInstance(failure, supervisor.HTTPException)

    def test_an_unlabelled_or_contradictory_first_delta_still_needs_a_second(self) -> None:
        for first in (delta(2), delta(2, seq=0, frames=3, phase="first"), delta(2, seq=1, frames=2, phase="steady")):
            chunks, headers, failure = self.relay([MODE, first, delta(8), generation(10), *DONE])
            self.assertIsNone(failure)
            self.assertEqual(chunks, [FRAME * 2, FRAME * 8])
            self.assertEqual(headers["x-tts-native-engine-chunk-proof"], "two-distinct-sse-delta-events")
        _chunks, _headers, failure = self.relay([MODE, delta(2), generation(2), *DONE])
        self.assertIsInstance(failure, supervisor.HTTPException)

    def test_low_latency_first_block_outlasts_the_wait_for_the_second_block(self) -> None:
        # With headers released on the first block, a 2-frame first block left
        # a player that starts at the first byte without audio for about 70 ms
        # (measured on the GPU with the 1.7B model); 3 frames do not.
        low = supervisor._default_tuning_profiles()["profiles"]["low-latency"]
        self.assertEqual((low["id"], low["name"]), ("low-latency", "Low Latency"))
        self.assertEqual(low["first_block_frames"], 3)
        self.assertEqual(low["steady_block_frames"], 8)
        self.assertEqual(low["revision"], 3)

    def test_saved_documents_pick_up_the_new_built_in_and_keep_custom_profiles(self) -> None:
        provider = supervisor.TUNING_PROVIDER
        old_low = {**supervisor._default_tuning_profiles()["profiles"]["low-latency"],
                   "revision": 2, "first_block_frames": 2}
        custom = {**old_low, "id": "mine", "name": "Mine", "revision": 1}
        document = supervisor._normalize_tuning_document({
            "version": 2,
            "profiles": {"low-latency": old_low, "mine": custom},
            "selections": {"realtime": {provider: "low-latency"}, "voice-studio": {provider: "mine"}},
        })
        # The saved selection still names the built-in and now resolves to 3 frames.
        self.assertEqual(document["selections"]["realtime"][provider], "low-latency")
        self.assertEqual(document["profiles"]["low-latency"]["first_block_frames"], 3)
        self.assertEqual(document["profiles"]["low-latency"]["revision"], 3)
        # A custom profile cloned from the old built-in keeps what its owner saved.
        self.assertEqual(document["selections"]["voice-studio"][provider], "mine")
        self.assertEqual(document["profiles"]["mine"]["first_block_frames"], 2)

    def test_the_pin_is_recorded_at_admission_and_never_forwarded(self) -> None:
        source = SUPERVISOR.read_text(encoding="utf-8")
        self.assertIn('payload["_lifecycle_pair_supplied"] = expected_epoch is not None', source)
        self.assertIn('None if payload.get("_lifecycle_pair_supplied") else {}', source)
        payload = {"_lifecycle_pair_supplied": True, "input": "x"}
        supervisor._strip_private_engine_fields(payload)
        self.assertEqual(payload, {"input": "x"})


if __name__ == "__main__":
    unittest.main()
