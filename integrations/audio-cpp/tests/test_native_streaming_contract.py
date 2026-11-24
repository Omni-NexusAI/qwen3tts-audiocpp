"""Regression checks for the reproducible native/offline Qwen3-TTS patch.

The package is one thirteen-file patch based on a pinned upstream tree.  The
optional parity test is enabled by ``AUDIO_CPP_AUTHORITATIVE_TREE`` during
release validation; it intentionally does not require CUDA or model assets.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

PATCH = Path(__file__).parents[1] / "patches" / "qwen3-native-pcm-streaming.patch"
DOCKERFILE = Path(__file__).parents[1] / "Dockerfile"
PINNED_REV = "238ab6a9e321c17de8e120559f57efeedaeb1345"
TOUCHED_FILES = (
    "app/server/http.cpp",
    "app/server/runtime.cpp",
    "include/engine/framework/runtime/cancellation.h",
    "include/engine/models/qwen3_tts/session.h",
    "include/engine/models/qwen3_tts/talker.h",
    "include/engine/models/qwen3_tts/tokenizer_speech_decoder.h",
    "include/engine/models/qwen3_tts/types.h",
    "src/models/qwen3_tts/loader.cpp",
    "src/models/qwen3_tts/session.cpp",
    "src/models/qwen3_tts/speaker_encoder.cpp",
    "src/models/qwen3_tts/talker.cpp",
    "src/models/qwen3_tts/tokenizer_speech_decoder.cpp",
    "src/models/qwen3_tts/tokenizer_speech_encoder.cpp",
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

    def test_default_stream_decoder_reproduces_the_offline_decode(self) -> None:
        native = self.patch.split("runtime::TaskResult Qwen3TTSSession::run_streaming_base_request", 1)[1]
        native = native.split("runtime::TaskResult Qwen3TTSSession::run(", 1)[0]
        # Default: each block is the offline decode of the same codes.  The
        # session hands over reference + generated codes, as the offline path
        # does, and the decoder mirrors decode()'s 300-frame chunk windows.
        self.assertIn(
            "audio = speech_decoder_->decode_streaming_block(accumulated, pending.frames, max_block_frames);",
            native,
        )
        self.assertIn("return Qwen3StreamDecoder::OfflineMatched;", self.patch)
        self.assertIn('"qwen3_tts.stream_decoder must be offline_matched or windowed"', self.patch)
        self.assertIn("const int64_t chunk_begin = (begin / kChunkCodes) * kChunkCodes;", self.patch)
        self.assertIn(
            "const int64_t chunk_context = chunk_begin > kLeftContextCodes ? kLeftContextCodes : chunk_begin;",
            self.patch,
        )
        # The transformer stage covers the whole offline window; only the
        # convolutional vocoder stage is limited to new frames plus its exact
        # look-back, which is computed from the loaded kernels.
        self.assertIn("DecoderGraphPart::Transformer", self.patch)
        self.assertIn("DecoderGraphPart::Vocoder", self.patch)
        self.assertIn("int64_t vocoder_lookback_samples(", self.patch)
        self.assertIn(
            "const int64_t vocoder_begin = std::max<int64_t>(window_begin, begin - vocoder_context);",
            self.patch,
        )
        # A constant cache replays one fixed tensor sequence, so the split
        # graphs must not share the complete graph's cache.
        self.assertIn("transformer_constants_", self.patch)
        self.assertIn("vocoder_constants_", self.patch)
        # The previous windowed decoder stays selectable per request.
        self.assertLess(
            native.index("if (stream_decoder == Qwen3StreamDecoder::OfflineMatched)"),
            native.index("audio = speech_decoder_->decode_padded(decoder_input, decoder_capacity_frames);"),
        )
        self.assertIn('"stream_decoder"', self.patch)

    def test_windowed_fixed_tail_and_reference_trim_are_packaged(self) -> None:
        native = self.patch.split("runtime::TaskResult Qwen3TTSSession::run_streaming_base_request", 1)[1]
        native = native.split("runtime::TaskResult Qwen3TTSSession::run(", 1)[0]
        self.assertIn(
            'parse_block_frames("qwen3_tts.stream_left_context_frames", 25)',
            self.patch,
        )
        self.assertIn("decoder_capacity_frames = left_context_frames + std::max(", native)
        self.assertIn("First, steady, and EOS-tail blocks use one padded causal decoder", native)
        self.assertNotIn("decode_and_trim_reference(*voice_prompt.reference_codes, pending)", native)
        self.assertIn("audio = speech_decoder_->decode_padded(decoder_input, decoder_capacity_frames);", native)
        self.assertIn("const int64_t drop_samples = context_frames * 1920;", native)
        self.assertIn("exact context samples are discarded", native.lower())
        self.assertIn("decode_padded(", self.patch)
        self.assertIn("std::vector<int32_t> padded_codes", self.patch)
        self.assertIn("graph_->run(padded_codes.data(), padded_codes.size())", self.patch)
        self.assertIn("codec_codes.frames * kDecodeSamplesPerCode", self.patch)

    def test_stream_sse_proof_is_engine_owned_and_precedes_pcm(self) -> None:
        self.assertIn("qwen3_stream_decode_mode_artifact", self.patch)
        self.assertIn('attach_decode_mode(event, "native-incremental-pcm");', self.patch)
        # The C++ JSON literal is escaped inside the unified patch.
        self.assertIn('\\"type\\":\\"speech.decode_mode\\"', self.patch)
        self.assertIn("streaming Qwen3 speech produced no decoder-mode proof", self.patch)
        self.assertIn('model.config.family == "qwen3_tts"', self.patch)
        self.assertIn("if (require_qwen_decode_mode && !wrote_decode_mode)", self.patch)
        self.assertIn("speech.audio.done", self.patch)

    def test_generic_speech_sse_does_not_require_qwen_artifacts(self) -> None:
        self.assertIn("const bool require_qwen_decode_mode", self.patch)
        self.assertIn("if (require_qwen_decode_mode && !wrote_decode_mode)", self.patch)
        self.assertIn("if (!wrote_audio)", self.patch)
        self.assertIn("if (require_qwen_decode_mode) {", self.patch)

    def test_windowed_decoder_context_is_upstream_overlap_not_model_reach(self) -> None:
        # 25 frames is upstream's chunk overlap.  The patch must not describe
        # it as the decoder's receptive field: the transformer attends 72
        # frames back per layer.
        self.assertIn("72 frames back in each of its layers", self.patch)
        self.assertNotIn("causal receptive field is 25 complete codec frames", self.patch)
        self.assertIn('parse_block_frames("qwen3_tts.stream_left_context_frames", 25)', self.patch)
        self.assertIn("decoder_capacity_frames = left_context_frames + std::max(", self.patch)
        self.assertIn("must not exceed 300 codec frames", self.patch)

    def test_voice_prompt_cache_is_reused_without_claiming_phrase_decoder_state(self) -> None:
        native = self.patch.split("runtime::TaskResult Qwen3TTSSession::run_streaming_base_request", 1)[1]
        native = native.split("runtime::TaskResult Qwen3TTSSession::run(", 1)[0]
        self.assertIn("resolve_voice_prompt(*qwen_request.voice_clone, prompt_builder)", native)
        self.assertIn("voice_prompt_cache_", self.patch)
        self.assertIn("no decoder-state", native)
        self.assertIn("continuity between phrase requests", native)
        self.assertNotIn("static Qwen3", native)

    def test_clone_conditioning_cache_is_keyed_and_truthful(self) -> None:
        # The default cache owns immutable clone conditioning. Generated state
        # stays phrase-local even when the opt-in prefix prototype is present.
        self.assertIn("resolve_voice_prompt(*qwen_request.voice_clone, prompt_builder)", self.patch)
        self.assertIn("qwen3_tts.voice_prompt_ms", self.patch)
        self.assertNotIn("talker_clone_conditioning_equal", self.patch)

    def test_prompt_is_prefilled_in_one_batched_pass_without_a_kv_prefix_cache(self) -> None:
        # The option keeps its name for deployed configurations, but it now
        # only sizes the host-side prompt material cache.
        self.assertIn('"qwen3_tts.talker_prefix_cache_slots"', self.patch)
        self.assertIn("talker_prefix_cache_slots_(mem_saver_from_options(options) ? 0", self.patch)
        self.assertIn("prompt_material_cache_slots = 0", self.patch)
        self.assertIn("voice_clone_prompt_material_cache_(prompt_material_cache_slots)", self.patch)
        self.assertIn("One batched pass over the whole prompt.", self.patch)
        self.assertIn("no talker KV state is cached", self.patch)
        # The one-row-at-a-time replay and the KV prefix cache are gone.
        for removed in (
            "immutable_prefix_cache_",
            "ImmutableTalkerPrefixCacheKey",
            "immutable_prefix_steps",
            "cached_prefill_steps",
            "current = cached_step_graph_->run_step(row_at(state.prompt, row, config.hidden_size));",
            "suffix_replay_ms",
        ):
            self.assertNotIn(removed, self.patch)

    def test_generated_speech_state_stays_request_local(self) -> None:
        self.assertIn("Never export\n+        // the cache after generated frames", self.patch)
        self.assertIn("must remain request-local", self.patch)
        self.assertIn("const runtime::TransformerKVState * cached_state = &prefill_output.state;", self.patch)
        self.assertIn("cached_graph_has_state = false", self.patch)

    def test_prompt_stage_timings_reach_the_event_and_the_headers(self) -> None:
        for field in ("prompt_rows", "prompt_embed_ms", "prompt_prefill_ms", "prompt_import_ms", "first_frame_ms"):
            self.assertIn(f'artifact.meta.emplace("{field}"', self.patch)
            self.assertIn(f'\\"{field}\\":', self.patch)
        for header in (
            "X-AudioCPP-Prompt-Ms",
            "X-AudioCPP-Prefill-Ms",
            "X-AudioCPP-Talker-Ms",
            "X-AudioCPP-Decode-Ms",
            "X-AudioCPP-Prompt-Rows",
            "X-AudioCPP-Prompt-Embed-Ms",
            "X-AudioCPP-Prompt-Prefill-Ms",
            "X-AudioCPP-Prompt-Import-Ms",
            "X-AudioCPP-First-Frame-Ms",
        ):
            self.assertIn(header, self.patch)

    def test_text_projection_cost_does_not_grow_ten_ms_per_text_token(self) -> None:
        # Upstream converted the whole projection weight on every call and
        # summed one scalar chain per output: about 10 ms per text token on
        # the 1.7B model, so first audio grew with the length of the text
        # (and looked like it grew with the generation cap, which the
        # supervisor derives from the text).
        talker = self.patch.split("b/src/models/qwen3_tts/talker.cpp", 1)[1].split("diff --git ", 1)[0]
        self.assertIn("+HostLinearWeights make_host_linear(", talker)
        self.assertIn("+    weights.text_projection_fc1_host = make_host_linear(", talker)
        self.assertIn("+    weights.text_projection_fc2_host = make_host_linear(", talker)
        # Converted once at load (inside make_host_linear), not per call.
        convert = 'assets::tensor_data_to_f32("Qwen3 talker host linear weight"'
        self.assertEqual(talker.count(convert), 1)
        self.assertLess(talker.index("+HostLinearWeights make_host_linear("), talker.index(convert))
        self.assertLess(talker.index(convert), talker.index("+void linear_host_rows("))
        # One independent update per output keeps each output's summation
        # order (same values as the plain dot product) and vectorizes.
        self.assertIn("+                    y[out] += column[out] * x;", talker)
        self.assertIn("-                sum += weight_values[static_cast<size_t>(out * in_features + in)] *", talker)
        self.assertIn("+        out.prompt_embed_ms = engine::debug::elapsed_ms(prompt_state_start, prompt_state_end);", talker)

    def test_every_stream_block_carries_the_engines_own_label(self) -> None:
        # seq / frames / phase let a relay tell from the first block alone
        # that audio is produced while the talker is still generating.
        self.assertIn('constexpr const char * kStreamBlockArtifactId = "qwen3_tts.stream_block";', self.patch)
        self.assertIn('final_block ? "final" : emitted_blocks == 0 ? "first" : "steady"', self.patch)
        self.assertIn("emit_pending(false);", self.patch)
        # Only the EOS tail is final; a generation-cap exit emits no tail.
        eos = self.patch.index("if (codes.termination == Qwen3GenerationTermination::Eos) {")
        self.assertLess(eos, self.patch.index("emit_pending(true);"))
        self.assertEqual(self.patch.count("emit_pending(true);"), 1)
        self.assertIn('if (artifact.id != "qwen3_tts.stream_block") continue;', self.patch)
        self.assertIn('",\\"seq\\":" + seq->second + ",\\"frames\\":" + frames->second +', self.patch)
        self.assertIn('",\\"phase\\":" + json_quote(phase->second);', self.patch)

    def test_a_closed_connection_stops_generation_within_one_frame(self) -> None:
        http = self.patch.split("b/app/server/http.cpp", 1)[1].split("diff --git ", 1)[0]
        self.assertIn("+bool socket_peer_closed(SocketHandle socket) {", http)
        self.assertIn('+constexpr std::string_view kSpeechPath = "/v1/audio/speech";', http)
        self.assertIn("+        const engine::runtime::ScopedCancellationProbe cancel_when_peer_closes(", http)
        # Installed before the handler runs, so it covers the non-streamed
        # path (generated inside handle()) and the streamed body alike.
        self.assertLess(
            http.index("ScopedCancellationProbe cancel_when_peer_closes("),
            http.index(" const auto response = handler.handle(request);"),
        )
        talker = self.patch.split("b/src/models/qwen3_tts/talker.cpp", 1)[1].split("diff --git ", 1)[0]
        loop = talker.index(" for (int64_t step = 0; step < max_new_tokens; ++step) {")
        check = talker.index("+            if (runtime::cancellation_requested()) {")
        sample = talker.index("apply_main_talker_processors(logits, config, generated_first_codes, step, repetition_penalty);")
        self.assertLess(loop, check)
        self.assertLess(check, sample)
        self.assertIn("inline thread_local const CancellationProbe * active_cancellation_probe = nullptr;", self.patch)

    def test_streamed_speech_has_a_send_timeout(self) -> None:
        # A client that stops reading must fail the write, not pin the model.
        self.assertIn("+            if (speech_request) {\n+                set_send_timeout(socket.get(), LiveIngestLimits{}.send_timeout_ms);", self.patch)

    def test_graph_inputs_are_uploaded_before_every_run(self) -> None:
        # The graph allocator frees an input after its last consumer and reuses
        # the memory, so positions and the attention mask written once at build
        # time are garbage from the second run on.  That, not block windowing,
        # was the main reason streamed audio differed from the offline decode.
        decoder = self.patch.split("b/src/models/qwen3_tts/tokenizer_speech_decoder.cpp", 1)[1]
        decoder = decoder.split("diff --git ", 1)[0]
        self.assertIn("void upload_fixed_inputs() {", decoder)
        upload_codes = decoder.split("int64_t upload_codes(const int32_t * codes, size_t code_count) {", 1)[1]
        upload_codes = upload_codes.split("void compute() {", 1)[0]
        self.assertIn("upload_fixed_inputs();", upload_codes)
        self.assertIn("-        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));", decoder)
        encoder = self.patch.split("b/src/models/qwen3_tts/tokenizer_speech_encoder.cpp", 1)[1]
        self.assertIn("void upload_fixed_inputs() {", encoder)
        # Once when the graph is built and once in run().
        self.assertEqual(encoder.count("+        upload_fixed_inputs();"), 2)

    def test_speaker_encoder_graph_is_built_for_the_exact_reference_length(self) -> None:
        # Reusing a larger graph zero-pads the features into pooling that
        # averages over every graph frame, which changes the speaker embedding
        # of a reference once a longer one has been encoded.
        self.assertIn("frames_ == frames && backend_ == backend", self.patch)
        self.assertIn("-        return weights_.get() == &weights && frames_ >= frames", self.patch)

    def test_clone_prompt_material_cache_reuses_only_exact_static_conditioning(self) -> None:
        # Base-clone prompt text is fused into early reference-code rows, so a
        # whole-reference KV reuse would be incorrect. The packaged cache keeps
        # only host-side clone material and the already-safe leading KV prefix.
        self.assertIn("struct VoiceClonePromptMaterial", self.patch)
        self.assertIn("struct VoiceClonePromptMaterialKey", self.patch)
        self.assertIn("reference_audio_hash", self.patch)
        self.assertIn("reference_sample_count", self.patch)
        self.assertIn("reference_text == rhs->reference_text", self.patch)
        self.assertIn("speaker_embedding_equal(lhs.speaker_embedding, rhs.speaker_embedding)", self.patch)
        self.assertIn("voice_clone_prompt_material_cache_", self.patch)
        self.assertIn("build_voice_clone_prompt_state_from_material", self.patch)
        self.assertIn("cannot be reused without changing model", self.patch)
        self.assertIn("No talker KV state is", self.patch)

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
