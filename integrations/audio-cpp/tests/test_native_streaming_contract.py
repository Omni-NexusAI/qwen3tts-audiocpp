"""Regression checks for the reproducible native/offline Qwen3-TTS patch.

The package is one eight-file patch based on a pinned upstream tree.  The
optional parity test is enabled by ``AUDIO_CPP_AUTHORITATIVE_TREE`` during
release validation; it intentionally does not require CUDA or model assets.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PATCH = Path(__file__).parents[1] / "patches" / "qwen3-native-pcm-streaming.patch"
DOCKERFILE = Path(__file__).parents[1] / "Dockerfile"
PINNED_REV = "238ab6a9e321c17de8e120559f57efeedaeb1345"
TOUCHED_FILES = (
    "app/server/runtime.cpp",
    "include/engine/models/qwen3_tts/session.h",
    "include/engine/models/qwen3_tts/talker.h",
    "include/engine/models/qwen3_tts/tokenizer_speech_decoder.h",
    "src/models/qwen3_tts/loader.cpp",
    "src/models/qwen3_tts/session.cpp",
    "src/models/qwen3_tts/talker.cpp",
    "src/models/qwen3_tts/tokenizer_speech_decoder.cpp",
)


class NativeStreamingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patch = PATCH.read_text(encoding="utf-8")

    def test_one_authoritative_patch_covers_all_engine_sources(self) -> None:
        headers = [line for line in self.patch.splitlines() if line.startswith("diff --git ")]
        self.assertEqual(len(headers), len(TOUCHED_FILES))
        for path in TOUCHED_FILES:
            self.assertIn(f"a/{path} b/{path}", self.patch)
        self.assertNotIn("qwen3-native-pcm-streaming-cancel-fix.patch", DOCKERFILE.read_text(encoding="utf-8"))

    def test_callback_delivery_is_final_result_only(self) -> None:
        self.assertIn("policy.output = runtime::StreamingOutputKind::FinalResult;", self.patch)
        self.assertIn("return std::nullopt;", self.patch)
        # PullEvents would make app/streaming/streaming.cpp forward returned
        # events in addition to callback delivery.  There must be one direct
        # sink dispatch site in the engine patch, not a second queued path.
        self.assertEqual(self.patch.count("stream_event_sink_(event);"), 1)

    def test_talker_emits_only_complete_codec_frames(self) -> None:
        completed = self.patch.index("++out.generated_codes.frames;")
        callback = self.patch.index("on_generated_frame(generated_frame);")
        self.assertLess(completed, callback)

    def test_callback_abort_resets_without_flushing_a_tail(self) -> None:
        self.assertIn("Socket callbacks are permitted to abort", self.patch)
        self.assertIn("Do not decode/emit an EOS tail", self.patch)
        self.assertIn("(void) talker_step_->release_cached_step_graph();", self.patch)

    def test_offline_full_reuses_streaming_session_without_pcm_events(self) -> None:
        self.assertIn('constexpr const char * kDecodeModeOption = "qwen3_tts.decode_mode";', self.patch)
        self.assertIn('constexpr const char * kOfflineFullMode = "offline_full";', self.patch)
        self.assertIn("streaming_result_ = run_offline_request(request);", self.patch)
        self.assertIn("streaming_result_ = run_streaming_base_request(request);", self.patch)
        offline_branch = self.patch.index("if (decode_mode == Qwen3RequestDecodeMode::OfflineFull)")
        native_branch = self.patch.index("streaming_result_ = run_streaming_base_request(request);")
        callback = self.patch.index("stream_event_sink_(event);")
        self.assertLess(offline_branch, native_branch)
        self.assertGreater(callback, native_branch)
        self.assertEqual(self.patch.count("stream_event_sink_(event);"), 1)

    def test_offline_full_result_is_one_shot_and_truthfully_reported(self) -> None:
        self.assertIn("runtime::TaskResult result = std::move(*streaming_result_);", self.patch)
        self.assertIn("streaming_result_.reset();", self.patch)
        self.assertIn('attach_decode_mode(*streaming_result_, "offline-full-decoder");', self.patch)
        self.assertIn('attach_decode_mode(*streaming_result_, "native-incremental-pcm");', self.patch)
        self.assertIn('"X-AudioCPP-Qwen3-Decode-Mode"', self.patch)

    def test_unknown_qwen_decode_mode_fails_explicitly(self) -> None:
        self.assertIn(
            '"qwen3_tts.decode_mode must be native_incremental or offline_full"',
            self.patch,
        )

    def test_fixed_tail_and_reference_trim_are_packaged(self) -> None:
        self.assertIn("left_context_frames + steady_block_frames > 300", self.patch)
        self.assertIn(
            'parse_block_frames("qwen3_tts.stream_left_context_frames", 72)',
            self.patch,
        )
        self.assertIn("single 72+steady-frame decoder graph", self.patch)
        self.assertNotIn(
            'parse_block_frames("qwen3_tts.stream_left_context_frames", 25)',
            self.patch,
        )
        self.assertNotIn("single 25+steady-frame decoder graph", self.patch)
        self.assertIn("decode_padded(", self.patch)
        self.assertIn("reference_codes.frames * kDecodeSamplesPerCode", self.patch)

    def test_patch_applies_and_matches_authoritative_tree(self) -> None:
        authoritative = os.environ.get("AUDIO_CPP_AUTHORITATIVE_TREE")
        if not authoritative:
            self.skipTest("set AUDIO_CPP_AUTHORITATIVE_TREE for clean-tree package parity validation")
        source = Path(authoritative).resolve()
        self.assertTrue(source.is_dir(), source)
        with tempfile.TemporaryDirectory(prefix="audio-cpp-native-patch-") as temporary:
            clean = Path(temporary) / "clean"
            subprocess.run(["git", "clone", "--no-checkout", str(source), str(clean)], check=True)
            subprocess.run(["git", "-C", str(clean), "checkout", "--detach", PINNED_REV], check=True)
            subprocess.run(["git", "-C", str(clean), "apply", "--check", str(PATCH)], check=True)
            subprocess.run(["git", "-C", str(clean), "apply", str(PATCH)], check=True)
            for relative in TOUCHED_FILES:
                self.assertEqual(
                    (clean / relative).read_bytes(),
                    (source / relative).read_bytes(),
                    relative,
                )


if __name__ == "__main__":
    unittest.main()
