"""The supervisor relays the engine's stage timings without inventing or requiring them."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from fastapi import HTTPException

SUPERVISOR = Path(__file__).parents[1] / "supervisor.py"

spec = importlib.util.spec_from_file_location("candidate_supervisor_stage_telemetry_test", SUPERVISOR)
assert spec and spec.loader
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)

BASE_EVENT = {"generated_frames": 30, "generation_cap": 180, "termination": "eos"}


class EngineStageTelemetryTests(unittest.TestCase):
    def test_stream_event_stage_fields_are_relayed(self) -> None:
        fields = supervisor._generation_event_fields({
            **BASE_EVENT,
            "prompt_ms": 3.5, "prefill_ms": 2.0, "talker_ms": 1500.0, "decode_ms": 240.0,
            "prompt_rows": 231, "prompt_prefill_ms": 140.25, "prompt_import_ms": 31.5,
            "prompt_embed_ms": 21.75, "first_frame_ms": 190.0, "stream_decoder": "offline-matched",
        })
        self.assertEqual(fields["promptRows"], 231)
        self.assertEqual(fields["promptEmbedMs"], 21.75)
        self.assertEqual(fields["promptPrefillMs"], 140.25)
        self.assertEqual(fields["promptImportMs"], 31.5)
        self.assertEqual(fields["firstFrameMs"], 190.0)
        self.assertEqual(fields["streamDecoder"], "offline-matched")
        self.assertEqual(fields["talkerMs"], 1500.0)

    def test_an_older_engine_without_stage_fields_still_reports(self) -> None:
        fields = supervisor._generation_event_fields(dict(BASE_EVENT))
        self.assertEqual(fields, {"generatedFrames": 30, "generationCap": 180, "eos": True})

    def test_malformed_stage_values_are_rejected_not_relayed(self) -> None:
        for bad in ({"prompt_prefill_ms": -1}, {"first_frame_ms": float("nan")}, {"prompt_import_ms": True},
                    {"prompt_rows": -5}, {"prompt_rows": 1.5}):
            with self.subTest(bad=bad), self.assertRaises(HTTPException):
                supervisor._generation_event_fields({**BASE_EVENT, **bad})
        # An unknown decoder name is dropped rather than relayed as fact.
        fields = supervisor._generation_event_fields({**BASE_EVENT, "stream_decoder": "something-else"})
        self.assertNotIn("streamDecoder", fields)

    def test_completed_response_headers_carry_the_same_report(self) -> None:
        headers = {
            "X-AudioCPP-Generated-Frames": "30", "X-AudioCPP-Generation-Cap": "180",
            "X-AudioCPP-Prompt-Ms": "3.500000", "X-AudioCPP-Prefill-Ms": "2.000000",
            "X-AudioCPP-Talker-Ms": "1500.000000", "X-AudioCPP-Decode-Ms": "240.000000",
            "X-AudioCPP-Prompt-Rows": "231", "X-AudioCPP-Prompt-Prefill-Ms": "140.250000",
            "X-AudioCPP-Prompt-Import-Ms": "31.500000", "X-AudioCPP-First-Frame-Ms": "190.000000",
        }
        fields = supervisor._generation_event_fields(supervisor._completed_generation_event(headers, "eos"))
        self.assertEqual(fields["promptRows"], 231)
        self.assertEqual(fields["promptPrefillMs"], 140.25)
        self.assertEqual(fields["decodeMs"], 240.0)
        # The three original headers alone remain sufficient.
        minimal = {"X-AudioCPP-Generated-Frames": "30", "X-AudioCPP-Generation-Cap": "180"}
        self.assertEqual(
            supervisor._generation_event_fields(supervisor._completed_generation_event(minimal, "eos")),
            {"generatedFrames": 30, "generationCap": 180, "eos": True},
        )

    def test_stage_fields_are_stored_in_outcomes_and_relayed_as_headers(self) -> None:
        for key in ("promptRows", "promptEmbedMs", "promptPrefillMs", "promptImportMs", "firstFrameMs",
                    "streamDecoder"):
            self.assertIn(key, supervisor.ENGINE_STAGE_OUTCOME_KEYS)
        for header in ("X-AudioCPP-Generated-Frames", "X-AudioCPP-Termination", "X-AudioCPP-Prompt-Rows",
                       "X-AudioCPP-Prompt-Embed-Ms",
                       "X-AudioCPP-Prompt-Prefill-Ms", "X-AudioCPP-First-Frame-Ms", "X-AudioCPP-Talker-Ms"):
            self.assertIn(header, supervisor.ENGINE_RELAYED_HEADERS)


if __name__ == "__main__":
    unittest.main()
