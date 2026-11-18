"""Contract tests for the voice steadiness and latency benchmark.

They exercise the analysis, request shaping and reporting without a server.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
import wave
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - the benchmark is optional tooling
    np = None

TOOL = Path(__file__).parents[1] / "tools" / "voice_bench.py"
RATE = 24000


def load_tool():
    spec = importlib.util.spec_from_file_location("voice_bench_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tone(frequency: float, seconds: float, amplitude: float = 8000.0) -> bytes:
    time_axis = np.arange(int(RATE * seconds)) / RATE
    return np.round(amplitude * np.sin(2 * np.pi * frequency * time_axis)).astype("<i2").tobytes()


@unittest.skipIf(np is None, "numpy is required for the voice benchmark")
class VoiceBenchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = load_tool()

    def test_tool_only_reads_state_and_speaks(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        for forbidden in ("/switch", "/unload", "/control/gpu-guard", "/control/switch", "/v1/tuning/selection",
                          ".put(", ".patch(", ".delete("):
            self.assertNotIn(forbidden, source)
        posted = set(re.findall(r'self\.base \+ "(/[^"]+)"', source))
        self.assertEqual(posted, {"/v1/tuning/resolve", "/v1/audio/speech"})
        self.assertIn("never loads or unloads a model", source)

    def test_pitch_and_brightness_of_a_known_tone(self) -> None:
        pitch, centroid = self.bench.features(tone(150.0, 1.0), RATE)
        self.assertAlmostEqual(pitch, 150.0, delta=3.0)
        self.assertIsNotNone(centroid)
        brighter = self.bench.features(tone(300.0, 1.0), RATE)[1]
        self.assertGreater(brighter, centroid)

    def test_silence_and_short_clips_have_no_pitch(self) -> None:
        self.assertEqual(self.bench.features(bytes(RATE * 2), RATE), (None, None))
        self.assertEqual(self.bench.features(tone(150.0, 0.01), RATE), (None, None))

    def test_octave_outliers_do_not_move_the_median(self) -> None:
        mostly = tone(120.0, 0.8) + tone(360.0, 0.2)
        pitch, _ = self.bench.features(mostly, RATE)
        self.assertAlmostEqual(pitch, 120.0, delta=3.0)

    def test_segments_follow_a_pitch_change(self) -> None:
        parts = self.bench.segment_features(tone(110.0, 1.0) + tone(150.0, 1.0) + tone(190.0, 1.0), RATE)
        self.assertEqual(len(parts), 3)
        self.assertLess(parts[0]["pitch_hz"], parts[1]["pitch_hz"])
        self.assertLess(parts[1]["pitch_hz"], parts[2]["pitch_hz"])
        pitch_range, _ = self.bench.spread(part["pitch_hz"] for part in parts)
        self.assertAlmostEqual(pitch_range, 80.0, delta=6.0)

    def test_spread_median_and_percentile_ignore_missing_values(self) -> None:
        self.assertEqual(self.bench.spread([None, 100.0]), (None, None))
        value_range, deviation = self.bench.spread([100.0, None, 120.0, 110.0])
        self.assertEqual(value_range, 20.0)
        self.assertAlmostEqual(deviation, 8.165, places=2)
        self.assertEqual(self.bench.median([None, 3.0, 1.0, 2.0]), 2.0)
        self.assertIsNone(self.bench.median([None]))
        self.assertAlmostEqual(self.bench.percentile([1.0, 2.0, 3.0, 4.0, None], 0.95), 3.85)

    def test_block_boundaries_follow_first_then_steady_blocks(self) -> None:
        frame = self.bench.SAMPLES_PER_CODEC_FRAME
        self.assertEqual(self.bench.block_boundaries(4, 12, 30 * frame), [4 * frame, 16 * frame, 28 * frame])
        self.assertEqual(self.bench.block_boundaries(0, 12, 30 * frame), [])

    def test_identical_clips_are_reported_identical(self) -> None:
        clip = tone(140.0, 1.0)
        found = self.bench.pcm_difference(clip, clip, RATE, [9600])
        self.assertTrue(found["identical"])
        self.assertEqual(found["max_abs_difference"], 0)
        self.assertIsNone(found["edge_ratio"])
        self.assertIsNone(self.bench.pcm_difference(b"", clip, RATE))

    def test_difference_concentrated_at_block_edges_raises_the_edge_ratio(self) -> None:
        generator = np.random.default_rng(7)
        base = np.frombuffer(tone(140.0, 1.0), dtype="<i2").astype(np.float64)
        changed = base + generator.normal(0, 4, len(base))
        boundary = 9600
        changed[boundary - 100:boundary + 100] += 600
        streamed = np.clip(np.round(changed), -32768, 32767).astype("<i2").tobytes()
        found = self.bench.pcm_difference(streamed, base.astype("<i2").tobytes(), RATE, [boundary])
        self.assertFalse(found["identical"])
        self.assertGreater(found["edge_ratio"], 20.0)
        even = np.clip(np.round(base + generator.normal(0, 4, len(base))), -32768, 32767).astype("<i2").tobytes()
        spread_evenly = self.bench.pcm_difference(even, base.astype("<i2").tobytes(), RATE, [boundary])
        self.assertLess(spread_evenly["edge_ratio"], 1.5)

    def test_length_difference_is_reported(self) -> None:
        found = self.bench.pcm_difference(tone(140.0, 1.0), tone(140.0, 0.5), RATE)
        self.assertEqual(found["length_delta_samples"], RATE // 2)
        self.assertFalse(found["identical"])

    def test_talker_fit_separates_fixed_cost_from_per_frame_cost(self) -> None:
        frames = [25, 73, 87, 135]
        intercept, slope = self.bench.linear_fit(frames, [1800 + 32 * count for count in frames])
        self.assertAlmostEqual(intercept, 1800.0, delta=1.0)
        self.assertAlmostEqual(slope, 32.0, delta=0.1)
        self.assertIsNone(self.bench.linear_fit([10, 10, 20], [1, 2, 3]))

    def test_replies_have_equal_sentence_counts_and_no_short_or_numeric_sentences(self) -> None:
        self.assertGreaterEqual(len(self.bench.REPLIES), 8)
        self.assertGreaterEqual(len(self.bench.SEEDS), len(self.bench.REPLIES))
        for reply in self.bench.REPLIES:
            self.assertEqual(len(reply), 5)
            for sentence in reply:
                self.assertGreaterEqual(len(sentence.split()), 6, sentence)
                self.assertIsNone(re.search(r"\d", sentence), sentence)
                self.assertRegex(sentence, r"[.?!]$")
        for phrase in self.bench.SHORT_PHRASES:
            self.assertLessEqual(len(phrase.split()), 3, phrase)

    def test_arm_texts_match_each_request_shape(self) -> None:
        self.assertEqual(len(self.bench.arm_texts("sentences-stream", 0)), 5)
        whole = self.bench.arm_texts("one-request-stream", 0)
        self.assertEqual(whole, [" ".join(self.bench.REPLIES[0])])
        self.assertEqual(self.bench.arm_texts("short-stream", 2), [self.bench.SHORT_PHRASES[2]])
        self.assertEqual(self.bench.arm_reply_count("short-whole", 3), len(self.bench.SHORT_PHRASES))
        self.assertEqual(self.bench.arm_reply_count("sentences-whole", 3), 3)
        for first, second in self.bench.PAIRS:
            self.assertEqual(self.bench.ARMS[first]["split"], self.bench.ARMS[second]["split"])
            self.assertNotEqual(self.bench.ARMS[first]["delivery"], self.bench.ARMS[second]["delivery"])
        for name in self.bench.DEFAULT_ARMS:
            self.assertIn(name, self.bench.ARMS)

    def test_request_bodies(self) -> None:
        common = dict(model="qwen3-tts-1.7b-base-bf16", voice="clone:abc123", seed=42, language="English",
                      reply_id="run-arm-0")
        streamed = self.bench.build_request("sentences-stream", "Hello there.", profile="balanced", **common)
        self.assertEqual(streamed["response_format"], "pcm")
        self.assertIs(streamed["stream"], True)
        self.assertEqual(streamed["voice"], "clone:abc123")
        self.assertEqual(streamed["tuning"], {"provider": "qwen3tts-audiocpp", "scope": "realtime",
                                              "profile_id": "balanced", "overrides": {"seed": 42}})
        self.assertNotIn("seed", streamed)
        self.assertNotIn("continuation_id", streamed)

        whole = self.bench.build_request("sentences-whole", "Hello there.", profile=None, **common)
        self.assertIs(whole["stream"], False)
        self.assertEqual(whole["seed"], 42)
        self.assertNotIn("tuning", whole)

        chained = self.bench.build_request("continuation", "Hello there.", profile="balanced", **common)
        self.assertEqual(chained["continuation_id"], "run-arm-0")

        with self.assertRaises(ValueError):
            self.bench.build_request("emulated-continuation", "Hello there.", profile="balanced", **common)
        emulated = self.bench.build_request("emulated-continuation", "Hello there.", profile="balanced",
                                            reference=(b"RIFFdata", "Reference words."), **common)
        self.assertNotIn("voice", emulated)
        self.assertEqual(emulated["task_type"], "Base")
        self.assertEqual(base64.b64decode(emulated["ref_audio"]), b"RIFFdata")
        self.assertEqual(emulated["ref_text"], "Reference words.")

    def test_sampling_overrides_ride_in_the_tuning_object_and_keep_the_seed(self) -> None:
        overrides = self.bench.parse_overrides(["temperature=0.6", "top_k=40"])
        self.assertEqual(overrides, {"temperature": 0.6, "top_k": 40})
        common = dict(model="m", voice="clone:abc123", seed=42, language="English", reply_id="r")
        body = self.bench.build_request("sentences-stream", "Hello there.", profile="balanced",
                                        overrides={**overrides, "seed": 7}, **common)
        self.assertEqual(body["tuning"]["overrides"], {"temperature": 0.6, "top_k": 40, "seed": 42})
        with self.assertRaises(ValueError):
            self.bench.build_request("sentences-stream", "Hello there.", profile=None, overrides=overrides, **common)
        for bad in ("temperature", "speed=2", "temperature=warm", "temperature=true", "seed=5"):
            with self.assertRaises(ValueError):
                self.bench.parse_overrides([bad])
        self.assertEqual(self.bench.parse_overrides([]), {})

    def test_continuation_reference_keeps_recent_whole_phrases_within_budget(self) -> None:
        stereo = np.stack([np.frombuffer(tone(130.0, 1.0), dtype="<i2")] * 2, axis=1).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(RATE)
            writer.writeframes(stereo.tobytes())
        previous = [("Oldest phrase.", tone(140.0, 1.0)), ("Middle phrase.", tone(150.0, 1.0)),
                    ("Newest phrase.", tone(160.0, 1.0))]
        audio, transcript, used = self.bench.continuation_reference(
            buffer.getvalue(), " Reference words. ", previous, budget_seconds=2.5, gap_ms=100)
        self.assertEqual(transcript, "Reference words. Middle phrase. Newest phrase.")
        self.assertAlmostEqual(used, 2.0)
        with wave.open(io.BytesIO(audio), "rb") as reader:
            self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()), (1, 2, RATE))
            self.assertAlmostEqual(reader.getnframes() / RATE, 1.0 + 2 * (0.1 + 1.0), places=3)
        _, bare_transcript, bare_used = self.bench.continuation_reference(buffer.getvalue(), "Reference words.", [])
        self.assertEqual((bare_transcript, bare_used), ("Reference words.", 0.0))

    def test_reference_is_resampled_to_the_model_rate(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(bytes(16000 * 2))
        audio, _, _ = self.bench.continuation_reference(buffer.getvalue(), "Words.", [])
        with wave.open(io.BytesIO(audio), "rb") as reader:
            self.assertEqual(reader.getframerate(), RATE)
            self.assertEqual(reader.getnframes(), RATE)

    def test_report_summarises_results_without_a_server(self) -> None:
        with tempfile.TemporaryDirectory(prefix="voice-bench-test-") as temporary:
            out = Path(temporary)
            rows = []
            pitches = [120.0, 140.0, 160.0]
            for arm, delivery in (("sentences-stream", "stream"), ("sentences-whole", "whole")):
                for phrase, frequency in enumerate(pitches):
                    pcm = tone(frequency, 1.0)
                    relative = Path("audio") / arm / f"r0_p{phrase}.wav"
                    (out / relative).parent.mkdir(parents=True, exist_ok=True)
                    (out / relative).write_bytes(self.bench.wav_bytes(pcm, RATE))
                    rows.append({
                        "arm": arm, "reply": 0, "phrase": phrase, "seed": 314159, "text": "x", "chars": 1,
                        "delivery": delivery, "profile": "balanced", "status": 200, "error": None,
                        "headers": {"x-tts-delivery-mode": "native-incremental-pcm" if delivery == "stream"
                                    else "buffered-fallback"},
                        "first_audio_s": 2.0 + phrase, "total_s": 1.0, "rate": RATE, "wav": relative.as_posix(),
                        "audio_s": 1.0, "pitch_hz": frequency, "centroid_hz": 500.0 + frequency,
                        "outcome": {"generatedFrames": 25 + 40 * phrase, "talkerMs": 1800 + 32 * (25 + 40 * phrase),
                                    "promptMs": 4, "prefillMs": 5, "decodeMs": 200},
                    })
            rows.append({"arm": "sentences-stream", "reply": 1, "phrase": 0, "seed": 271828, "text": "x", "chars": 1,
                         "delivery": "stream", "profile": "balanced", "status": 502, "error": "refused",
                         "headers": {}, "total_s": 0.1, "rate": RATE})
            (out / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (out / "run.json").write_text(json.dumps({"model": "m", "voice": "clone:v", "profile": "balanced",
                                                      "first_block_frames": 4, "steady_block_frames": 12}),
                                          encoding="utf-8")
            code = self.bench.report(argparse.Namespace(out=str(out)))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)  # one request failed
        stream = summary["arms"]["sentences-stream"]
        self.assertEqual((stream["requests"], stream["failed"], stream["replies"]), (4, 1, 1))
        self.assertEqual(stream["failures"][0]["status"], 502)
        self.assertEqual(stream["request_pitch_range"], 40.0)
        self.assertAlmostEqual(stream["segment_pitch_range"], 40.0, delta=4.0)
        self.assertEqual(stream["first_audio_s_first_request"], 2.0)
        self.assertEqual(stream["first_audio_s_later_requests"], 3.5)
        self.assertAlmostEqual(stream["talker_fixed_ms"], 1800.0, delta=1.0)
        self.assertEqual(stream["delivery_modes"], {"native-incremental-pcm": 3})
        self.assertEqual(summary["arms"]["sentences-whole"]["failed"], 0)
        pair = summary["pairs"]["sentences-stream vs sentences-whole"]
        self.assertEqual((pair["compared"], pair["identical"], pair["length_mismatches"]), (3, 3, 0))


class _StubSpeechServer:
    """A tiny stand-in for the speech server: fixed tone audio, no engine."""

    def __init__(self, *, loaded: bool = True) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        stub = self
        self.loaded = loaded
        self.requests: list[tuple[str, str, dict]] = []
        reference = io.BytesIO()
        with wave.open(reference, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(RATE)
            writer.writeframes(tone(125.0, 0.5))
        self.reference = reference.getvalue()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                return

            def _json(self, payload, status: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                stub.requests.append(("GET", self.path, {}))
                if self.path == "/control/status":
                    self._json({"state": "loaded" if stub.loaded else "unloaded",
                                "activeModel": "qwen3-tts-1.7b-base-bf16" if stub.loaded else None, "engineEpoch": 4})
                elif self.path == "/v1/voices/profiles":
                    self._json({"selectedVoice": "clone:abc123", "defaultVoice": "clone:abc123", "voices": []})
                elif self.path == "/v1/voices/profiles/abc123":
                    self._json({"id": "abc123", "ref_audio": base64.b64encode(stub.reference).decode("ascii"),
                                "ref_text": "Reference words."})
                elif self.path.startswith("/v1/audio/outcomes/"):
                    self._json({"state": "completed", "generatedFrames": 25, "talkerMs": 2600, "promptMs": 4,
                                "prefillMs": 5, "decodeMs": 200})
                else:
                    self._json({"detail": "not found"}, 404)

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                stub.requests.append(("POST", self.path, body))
                if self.path == "/v1/tuning/resolve":
                    self._json({"profile": {"id": "balanced", "revision": 2, "first_block_frames": 4,
                                            "steady_block_frames": 12}, "temporaryOverrides": {}})
                elif self.path == "/v1/audio/speech":
                    if body.get("input") == "No.":
                        self._json({"detail": "audio.cpp generation completed before two native PCM events"}, 502)
                        return
                    pcm = tone(120.0 + 10 * (len(body["input"]) % 5), 0.6)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/pcm")
                    self.send_header("Content-Length", str(len(pcm)))
                    self.send_header("X-TTS-Sample-Rate", str(RATE))
                    self.send_header("X-TTS-Request-Id", "0" * 32)
                    self.send_header("X-TTS-Delivery-Mode",
                                     "native-incremental-pcm" if body.get("stream") else "buffered-fallback")
                    self.end_headers()
                    self.wfile.write(pcm)
                else:
                    self._json({"detail": "not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "_StubSpeechServer":
        import threading

        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def _run_arguments(base_url: str, out: Path, arms: str, **changes) -> argparse.Namespace:
    values = dict(base_url=base_url, arms=arms, replies=2, voice="", profile="balanced", language="English",
                  continuation_seconds=12.0, no_warmup=False, out=str(out), override=None)
    values.update(changes)
    return argparse.Namespace(**values)


@unittest.skipIf(np is None or importlib.util.find_spec("httpx") is None, "numpy and httpx are required")
class VoiceBenchRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = load_tool()

    def test_run_refuses_when_no_model_is_loaded(self) -> None:
        with _StubSpeechServer(loaded=False) as stub, tempfile.TemporaryDirectory() as temporary:
            code = self.bench.run(_run_arguments(stub.url, Path(temporary), "sentences-stream"))
            self.assertEqual(code, 3)
            self.assertEqual([item[:2] for item in stub.requests], [("GET", "/control/status")])

    def test_run_speaks_each_arm_and_reports(self) -> None:
        with _StubSpeechServer() as stub, tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            code = self.bench.run(_run_arguments(
                stub.url, out, "sentences-stream,sentences-whole,one-request-stream,emulated-continuation"))
            self.assertEqual(code, 0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            rows = self.bench.load_rows(out)
            again = self.bench.run(_run_arguments(stub.url, out, "sentences-stream", no_warmup=True))
            self.assertEqual(again, 0)
            self.assertEqual(len(self.bench.load_rows(out)), len(rows))  # finished phrases are not repeated
        self.assertEqual({item[1] for item in stub.requests if item[0] == "POST"},
                         {"/v1/tuning/resolve", "/v1/audio/speech"})
        self.assertEqual(summary["arms"]["sentences-stream"]["requests"], 10)
        self.assertEqual(summary["arms"]["one-request-stream"]["requests"], 2)
        self.assertEqual(summary["arms"]["sentences-stream"]["failed"], 0)
        self.assertEqual(summary["arms"]["sentences-stream"]["delivery_modes"], {"native-incremental-pcm": 10})
        self.assertEqual(summary["pairs"]["sentences-stream vs sentences-whole"]["identical"], 10)
        self.assertEqual(summary["run"]["first_block_frames"], 4)
        speech = [item[2] for item in stub.requests if item[1] == "/v1/audio/speech"]
        warmups = [body for body in speech if body["input"] == self.bench.WARMUP_TEXT]
        self.assertEqual(len(warmups), 4)  # one per arm, never measured
        self.assertNotIn(self.bench.WARMUP_TEXT, [row["text"] for row in rows])
        emulated = [row for row in rows if row["arm"] == "emulated-continuation"]
        self.assertEqual([row["continuation_seconds_used"] for row in emulated if row["reply"] == 0],
                         [0.0, 0.6, 1.2, 1.8, 2.4])
        chained = [body for body in speech if body.get("task_type") == "Base" and body["input"] != self.bench.WARMUP_TEXT]
        self.assertTrue(chained[1]["ref_text"].startswith("Reference words. "))
        self.assertNotIn("voice", chained[1])
        self.assertEqual({body["tuning"]["overrides"]["seed"] for body in speech[1:6]}, {self.bench.SEEDS[0]})

    def test_a_sweep_run_sends_its_override_and_does_not_mix_into_another_run(self) -> None:
        with _StubSpeechServer() as stub, tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            code = self.bench.run(_run_arguments(stub.url, out, "one-request-stream", replies=1, no_warmup=True,
                                                 override=["temperature=0.6"]))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            mixed = self.bench.run(_run_arguments(stub.url, out, "one-request-stream", replies=1, no_warmup=True,
                                                  override=["temperature=0.9"]))
            same = self.bench.run(_run_arguments(stub.url, out, "one-request-stream", replies=1, no_warmup=True,
                                                 override=["temperature=0.6"]))
            bare = self.bench.run(_run_arguments(stub.url, out / "bare", "one-request-stream", replies=1,
                                                 profile="none", override=["temperature=0.6"]))
            unknown = self.bench.run(_run_arguments(stub.url, out / "bad", "one-request-stream", replies=1,
                                                    override=["speed=2"]))
        self.assertEqual((code, mixed, same, bare, unknown), (0, 2, 0, 2, 2))
        self.assertEqual(summary["run"]["overrides"], {"temperature": 0.6})
        speech = [item[2] for item in stub.requests if item[1] == "/v1/audio/speech"]
        self.assertEqual(len(speech), 1)  # the refused runs sent nothing and the repeat run had nothing left to do
        self.assertEqual(speech[0]["tuning"]["overrides"], {"temperature": 0.6, "seed": self.bench.SEEDS[0]})

    def test_run_reports_a_refused_short_phrase_as_a_failure(self) -> None:
        with _StubSpeechServer() as stub, tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            code = self.bench.run(_run_arguments(stub.url, out, "short-stream", no_warmup=True))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        arm = summary["arms"]["short-stream"]
        self.assertEqual((arm["requests"], arm["failed"]), (len(self.bench.SHORT_PHRASES), 1))
        self.assertEqual(arm["failures"][0]["status"], 502)
        self.assertIn("two native PCM events", arm["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
