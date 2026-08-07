"""Focused contracts for Voice Studio mode separation and continuous playback."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest


ROOT = Path(__file__).parents[3]
INTEGRATION = Path(__file__).parents[1]
UI = INTEGRATION / "gradio_voice_studio.py"
WORKLET = ROOT / "web" / "hf-realtime-voice" / "worklets" / "studio-playback.js"
NODE_TEST = Path(__file__).with_name("studio_playback.test.mjs")

gradio_stub = types.ModuleType("gradio")
gradio_stub.Blocks = object
gradio_stub.themes = types.SimpleNamespace(Base=object)
sys.modules.setdefault("gradio", gradio_stub)

ui_spec = importlib.util.spec_from_file_location("candidate_studio_playback_test", UI)
assert ui_spec and ui_spec.loader
studio = importlib.util.module_from_spec(ui_spec)
ui_spec.loader.exec_module(studio)


class StudioPlaybackContractTests(unittest.TestCase):
    def test_full_quality_has_dedicated_offline_policy_and_preserves_format(self) -> None:
        payload = {
            "input": "hello",
            "response_format": "pcm",
            "stream": True,
            "tuning": {"profile_id": "low-latency", "overrides": {"temperature": 1.4}},
        }
        returned = studio.apply_full_wav_quality_policy(payload, "flac")
        self.assertIs(returned, payload)
        self.assertEqual(payload["response_format"], "flac")
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["tuning"],
            {
                "provider": "qwen3tts-audiocpp",
                "scope": "voice-studio",
                "profile_id": "quality",
                "overrides": {},
            },
        )
        with self.assertRaises(ValueError):
            studio.apply_full_wav_quality_policy({}, "streaming-wav")

    def test_full_wav_callback_cannot_inherit_streaming_session_state(self) -> None:
        source = UI.read_text(encoding="utf-8")
        callback = source.split("def on_play_generate(", 1)[1].split("# Streaming mode callbacks", 1)[0]
        self.assertIn("apply_full_wav_quality_policy(payload, response_format)", callback)
        self.assertNotIn("apply_session_tuning(", callback)
        self.assertNotIn("request_tts_streaming(", callback)
        self.assertIn("offline-full-decoder", callback)
        self.assertIn("Quality (dedicated Full Quality)", callback)
        wiring = source.split("play_generate_btn.click(", 1)[1].split("# Streaming mode wiring", 1)[0]
        self.assertNotIn("tuning_profile_dropdown", wiring)
        self.assertNotIn("tuning_override_state", wiring)
        self.assertIn("play_response_format = gr.Dropdown", source)
        for output_format in ("wav", "pcm", "flac", "mp3", "aac", "opus"):
            self.assertIn(f'"{output_format}")', source)
        self.assertIn("play_response_format, play_speed", wiring)

    def test_base_clone_preview_is_offline_only_without_streaming_fallback(self) -> None:
        source = UI.read_text(encoding="utf-8")
        callback = source.split("def on_generate_clone(", 1)[1].split("def on_save_clone_profile", 1)[0]
        self.assertIn('apply_full_wav_quality_policy(payload, "wav")', callback)
        self.assertIn("request_tts_voice_clone(", callback)
        self.assertNotIn("request_tts_streaming(", callback)
        self.assertNotIn("fallback", callback.lower())

    def test_active_widget_uses_profile_phrase_policy_and_continuous_worklet(self) -> None:
        source = UI.read_text(encoding="utf-8")
        active_widget = source.rsplit("def _build_streaming_widget_html(", 1)[1].split("# Callback implementations", 1)[0]
        self.assertNotIn("getPhraseSettings", active_widget)
        self.assertNotIn("phrase-min", active_widget)
        self.assertNotIn("phrase-max", active_widget)
        self.assertNotIn("phrase-idle", active_widget)
        self.assertNotIn("createBufferSource", active_widget)
        self.assertIn("findImmediateBoundary", active_widget)
        self.assertIn("findSafePhraseCut", active_widget)
        self.assertIn("config.textLookahead", active_widget)
        self.assertIn("config.phraseFlushMs", active_widget)
        self.assertIn("config.phraseHardCap", active_widget)
        self.assertIn("ensurePlaybackQueue", active_widget)
        self.assertIn("enqueuePcmForTurn", active_widget)
        self.assertIn("finishPlaybackTurn", active_widget)
        self.assertIn("kind: 'clear'", active_widget)
        self.assertIn("config.playbackStartupMs", active_widget)

    def test_worklet_is_packaged_by_both_candidate_images(self) -> None:
        self.assertTrue(WORKLET.is_file())
        dockerfile = (INTEGRATION / "Dockerfile").read_text(encoding="utf-8")
        overlay = (INTEGRATION / "Dockerfile.overlay").read_text(encoding="utf-8")
        self.assertIn("COPY web/hf-realtime-voice/ .", dockerfile)
        self.assertIn("COPY web/hf-realtime-voice/ /opt/voice-studio/", overlay)
        self.assertIn(studio.STUDIO_PLAYBACK_WORKLET_URL.split("?", 1)[0], UI.read_text(encoding="utf-8"))

    def test_streaming_controls_are_disabled_for_full_wav(self) -> None:
        source = UI.read_text(encoding="utf-8")
        callback = source.split("def on_play_mode_change", 1)[1].split("def on_s_voice_change", 1)[0]
        self.assertIn("is_streaming = mode != FULL_WAV_PLAYBACK_MODE", callback)
        self.assertIn("gr.update(interactive=is_native)", callback)
        self.assertIn("gr.update(interactive=is_streaming)", callback)
        self.assertIn("Full Quality is independent", callback)
        self.assertIn("All streaming and tuning-profile controls are locked", callback)
        self.assertIn("tuning_context_unlock", source)
        self.assertIn("first_block_frames", source.split("def resolve_tuning_override", 1)[1])
        self.assertIn("max_reference_seconds", source.split("def resolve_tuning_override", 1)[1])
        self.assertIn('profile.get("left_context_frames", 72)', source)
        self.assertIn('value=72', source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for AudioWorklet behavior tests")
    def test_audio_worklet_behavior(self) -> None:
        completed = subprocess.run(
            [shutil.which("node"), str(NODE_TEST)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("studio playback worklet tests passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
