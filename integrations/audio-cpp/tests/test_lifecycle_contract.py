"""Contract tests for load on demand and idle unload.

The supervisor is loaded against a private voice directory with a stub engine
and a fake clock, so no GPU, engine binary or real waiting is involved.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

INTEGRATION = Path(__file__).parents[1]
SUPERVISOR = INTEGRATION / "supervisor.py"
TEMPLATE = INTEGRATION / "qwen3-tts-base-f16.json"
LARGE = "qwen3-tts-1.7b-base-bf16"
SMALL = "qwen3-tts-0.6b-base-bf16"
LEGACY_NOT_LOADED = "No candidate model is loaded. Use Voice Studio model controls first."
LEGACY_NOT_ACTIVE = "Selected model is not active; switch it in Voice Studio before synthesis."

_VOICES = tempfile.TemporaryDirectory(prefix="lifecycle-voices-")


def _load_supervisor():
    environment = {
        "VOICE_LIBRARY_DIR": _VOICES.name,
        "AUDIO_CPP_CONFIG_TEMPLATE": str(TEMPLATE),
        "AUDIO_CPP_NATIVE_INCREMENTAL_PCM": "false",
        "AUDIO_CPP_NATIVE_LOAD_WARMUP": "false",
    }
    for name in ("AUDIO_CPP_IDLE_UNLOAD_SECONDS", "AUDIO_CPP_KEEP_LOADED", "AUDIO_CPP_SPEECH_WHEN_UNLOADED",
                 "AUDIO_CPP_LOAD_WAIT_SECONDS"):
        os.environ.pop(name, None)
    with patch.dict(os.environ, environment):
        spec = importlib.util.spec_from_file_location("candidate_supervisor_lifecycle_test", SUPERVISOR)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


supervisor = _load_supervisor()
REAL_GPU_GUARD = supervisor.gpu_guard


class _FakeProcess:
    returncode = None

    def poll(self):
        return None


class _LifecycleFixture(unittest.TestCase):
    """A stub engine, an always-open GPU guard and a fake clock around the supervisor."""

    def setUp(self) -> None:
        self.clock = 1000.0
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.guard = {"ok": True, "freeMiB": 12000}
        self.gate: asyncio.Event | None = None
        self.fail_start = False
        test = self

        def start_engine(model_id: str) -> None:
            if test.fail_start:
                raise RuntimeError("engine binary is missing")
            test.started.append(model_id)
            supervisor.engine = _FakeProcess()

        def stop_engine(*, action: str = "release") -> None:
            test.stopped.append(action)
            supervisor.engine = None

        async def engine_ready() -> bool:
            if supervisor.engine is None:
                return False
            if test.gate is not None:
                await test.gate.wait()
            return True

        self._patches = [
            patch.object(supervisor, "_start_engine", start_engine),
            patch.object(supervisor, "_stop_engine", stop_engine),
            patch.object(supervisor, "_engine_ready", engine_ready),
            patch.object(supervisor, "gpu_guard", lambda *_args, **_kwargs: dict(test.guard)),
            patch.object(supervisor, "_lifecycle_now", lambda: test.clock),
            patch.object(supervisor, "switch_lock", asyncio.Lock()),
            patch.object(supervisor, "generation_lock", asyncio.Lock()),
        ]
        for item in self._patches:
            item.start()
        supervisor.engine = None
        supervisor.EVENTS.clear()
        supervisor.state.update(activeModel=LARGE, state="unloaded", reason=None, lastError=None,
                                lastLoadElapsedS=None, engineEpoch=0)
        supervisor.lifecycle.update(last_activity=self.clock, manual_hold=False, last_unload_reason=None,
                                    load_task=None, load_model=None, load_started=None, load_started_at=None,
                                    last_load_error=None, load_failed_at=None, watch_task=None)
        supervisor.lifecycle_settings.clear()
        supervisor.lifecycle_settings.update(supervisor._default_lifecycle_settings())
        supervisor.lifecycle_settings["load_wait_seconds"] = 5

    def tearDown(self) -> None:
        for item in self._patches:
            item.stop()
        supervisor.engine = None

    async def _finish_load(self) -> None:
        task = supervisor.lifecycle["load_task"]
        if task is not None:
            await task


class LifecycleContractTests(_LifecycleFixture):
    # ------------------------------------------------------------------ ensure

    def test_defaults_are_idle_unload_after_fifteen_minutes_and_wait_for_speech(self) -> None:
        defaults = supervisor._default_lifecycle_settings()
        self.assertEqual(defaults, {"keep_loaded": False, "idle_unload_seconds": 900,
                                    "speech_when_unloaded": "wait", "load_wait_seconds": 90,
                                    "wake_on_health": False})

    def test_ensure_answers_at_once_loads_in_background_and_is_idempotent(self) -> None:
        async def scenario():
            self.gate = asyncio.Event()
            first = await supervisor.ensure_model(None)
            self.assertEqual(self.started, [])  # nothing has run yet: the answer did not wait
            await asyncio.sleep(0)
            second = await supervisor.ensure_model(LARGE)
            self.gate.set()
            await self._finish_load()
            third = await supervisor.ensure_model(LARGE)
            return first, second, third

        first, second, third = asyncio.run(scenario())
        self.assertEqual((first["model"], first["state"], first["ready"], first["accepted"]),
                         (LARGE, "loading", False, True))
        self.assertEqual(first["reason"], "load-in-progress")
        self.assertEqual(first["etaS"], 60)  # no measured load yet
        self.assertEqual((second["accepted"], second["ready"], second["reason"]), (True, False, "load-in-progress"))
        self.assertEqual(self.started, [LARGE])  # the second call did not start a second load
        self.assertEqual((third["state"], third["ready"], third["accepted"], third["reason"]),
                         ("loaded", True, True, None))
        self.assertIsNone(third["retryAfterS"])
        self.assertEqual(third["resident"], LARGE)
        self.assertEqual(third["engineEpoch"], 1)
        self.assertEqual(third["supervisorInstanceId"], supervisor.SUPERVISOR_INSTANCE_ID)
        for key in ("model", "state", "ready", "accepted", "reason", "retryable", "retryAfterS", "etaS",
                    "resident", "engineEpoch", "supervisorInstanceId", "gpuGuard"):
            self.assertIn(key, first)

    def test_ensure_says_why_when_the_gpu_guard_refuses(self) -> None:
        self.guard = {"ok": False, "reason": "GPU load blocked: 3000 MiB free, 10500 MiB required.", "freeMiB": 3000}
        answer = asyncio.run(supervisor.ensure_model(LARGE))
        self.assertEqual((answer["accepted"], answer["ready"], answer["reason"]), (False, False, "gpu-guard"))
        self.assertIn("3000 MiB free", answer["message"])
        self.assertEqual(answer["gpuGuard"]["freeMiB"], 3000)
        self.assertTrue(answer["retryable"])
        self.assertEqual(self.started, [])
        self.assertEqual(supervisor.state["state"], "unloaded")
        self.assertEqual(supervisor.EVENTS[-1]["action"], "gpu-blocked-load")

    def test_ensure_rejects_unknown_models_and_switch_values(self) -> None:
        with self.assertRaises(HTTPException) as unknown:
            asyncio.run(supervisor.ensure_model("not-a-model"))
        self.assertEqual(unknown.exception.status_code, 404)
        with self.assertRaises(HTTPException) as bad:
            asyncio.run(supervisor.ensure_model(LARGE, switch="always"))
        self.assertEqual(bad.exception.status_code, 422)

    def test_ensure_does_not_switch_away_from_a_recently_used_model(self) -> None:
        async def scenario():
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            soon = await supervisor.ensure_model(SMALL)
            never = await supervisor.ensure_model(SMALL, switch="never")
            self.clock += supervisor.SWITCH_GRACE_SECONDS + 1
            later = await supervisor.ensure_model(SMALL)
            await self._finish_load()
            return soon, never, later

        soon, never, later = asyncio.run(scenario())
        self.assertEqual((soon["accepted"], soon["reason"], soon["resident"]), (False, "resident-model-in-use", LARGE))
        self.assertTrue(soon["retryable"])
        self.assertGreaterEqual(soon["retryAfterS"], 100)
        self.assertEqual((never["accepted"], never["retryable"]), (False, False))
        self.assertEqual((later["accepted"], later["reason"]), (True, "load-in-progress"))
        self.assertEqual(self.started, [LARGE, SMALL])
        self.assertEqual(self.stopped, ["old-model-released"])  # the old model went first: one resident
        self.assertEqual(supervisor.state["activeModel"], SMALL)

    def test_ensure_never_interrupts_a_generation(self) -> None:
        async def scenario():
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            self.clock += supervisor.SWITCH_GRACE_SECONDS + 1
            async with supervisor.generation_lock:
                busy = await supervisor.ensure_model(SMALL)
                stopped_while_speaking = list(self.stopped)
            return busy, stopped_while_speaking

        busy, stopped_while_speaking = asyncio.run(scenario())
        self.assertEqual((busy["accepted"], busy["reason"]), (False, "resident-model-in-use"))
        self.assertEqual(stopped_while_speaking, [])

    def test_backend_ensure_route_accepts_an_empty_body(self) -> None:
        async def scenario():
            answer = await supervisor.backend_ensure(None)
            await self._finish_load()
            return answer

        answer = asyncio.run(scenario())
        self.assertEqual((answer["model"], answer["accepted"]), (LARGE, True))
        with self.assertRaises(HTTPException) as bad:
            asyncio.run(supervisor.backend_ensure({"model": 5}))
        self.assertEqual(bad.exception.status_code, 422)

    # ------------------------------------------------------------ manual hold

    def test_manual_unload_holds_until_the_user_loads_again(self) -> None:
        async def scenario():
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            unloaded = await supervisor.backend_unload()
            held = await supervisor.ensure_model(LARGE)
            try:
                await supervisor._assert_synthesis_ready({"model": LARGE})
                speech = None
            except HTTPException as exc:
                speech = exc
            manual = await supervisor.backend_switch({"model_key": LARGE, "wait": False})
            await self._finish_load()
            return unloaded, held, speech, manual

        unloaded, held, speech, manual = asyncio.run(scenario())
        self.assertEqual(unloaded["state"], "unloaded")
        self.assertTrue(unloaded["lifecycle"]["manualHold"])
        self.assertEqual(unloaded["lifecycle"]["lastUnloadReason"], "manual")
        self.assertEqual((held["accepted"], held["reason"], held["retryable"]), (False, "manual-hold", False))
        self.assertEqual((speech.status_code, speech.detail), (409, LEGACY_NOT_LOADED))
        self.assertEqual((manual["accepted"], manual["reason"]), (True, "load-in-progress"))
        self.assertFalse(supervisor.lifecycle["manual_hold"])
        self.assertEqual(supervisor.state["state"], "loaded")
        self.assertEqual(self.started, [LARGE, LARGE])
        self.assertEqual(self.stopped, ["explicit-unload"])

    def test_blocking_manual_load_still_works_and_clears_the_hold(self) -> None:
        async def scenario():
            supervisor.lifecycle["manual_hold"] = True
            return await supervisor.backend_switch({"model_key": LARGE})

        result = asyncio.run(scenario())
        self.assertEqual(result["state"], "loaded")
        self.assertTrue(result["singleResident"])
        self.assertFalse(supervisor.lifecycle["manual_hold"])

    # ------------------------------------------------------------ idle unload

    def test_idle_unload_waits_for_the_quiet_time_and_never_fires_during_use(self) -> None:
        async def scenario():
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            results = []
            self.clock += 899
            results.append(await supervisor._idle_tick())  # one second short
            async with supervisor.generation_lock:
                self.clock += 5000
                results.append(await supervisor._idle_tick())  # speaking: never
            results.append(await supervisor._idle_tick())  # just after use: the quiet time restarted
            self.clock += 899
            results.append(await supervisor._idle_tick())
            status_before = supervisor._lifecycle_status()
            self.clock += 1
            results.append(await supervisor._idle_tick())
            return results, status_before

        results, status_before = asyncio.run(scenario())
        self.assertEqual(results, [False, False, False, False, True])
        self.assertEqual(status_before["unloadInS"], 1.0)
        self.assertEqual(status_before["idleForS"], 899.0)
        self.assertEqual(self.stopped, ["idle-unload"])
        self.assertEqual(supervisor.state["state"], "unloaded")
        self.assertIn("900 seconds without speech", supervisor.state["reason"])
        status = supervisor._lifecycle_status()
        self.assertEqual((status["lastUnloadReason"], status["manualHold"]), ("idle", False))
        self.assertIsNone(status["unloadInS"])

    def test_speech_and_ensure_restart_the_quiet_time(self) -> None:
        async def scenario():
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            self.clock += 800
            await supervisor._assert_synthesis_ready({"model": LARGE})
            self.clock += 800
            first = await supervisor._idle_tick()
            await supervisor.ensure_model(LARGE)
            self.clock += 800
            second = await supervisor._idle_tick()
            payload: dict = {}
            request_id = supervisor._begin_request_outcome(payload, LARGE, "buffered-fallback")
            self.clock += 800
            supervisor._finish_request_outcome(request_id, "completed")
            self.clock += 800
            third = await supervisor._idle_tick()
            self.clock += 100
            fourth = await supervisor._idle_tick()
            return first, second, third, fourth

        self.assertEqual(asyncio.run(scenario()), (False, False, False, True))

    def test_keep_loaded_turns_idle_unload_off(self) -> None:
        async def scenario():
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            await supervisor.control_lifecycle_update({"keepLoaded": True})
            self.clock += 100000
            return await supervisor._idle_tick(), supervisor._lifecycle_status()

        fired, status = asyncio.run(scenario())
        self.assertFalse(fired)
        self.assertTrue(status["keepLoaded"])
        self.assertIsNone(status["unloadInS"])
        self.assertEqual(supervisor.state["state"], "loaded")

    def test_shortening_the_quiet_time_does_not_unload_at_once(self) -> None:
        async def scenario():
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            self.clock += 500
            await supervisor.control_lifecycle_update({"idleUnloadSeconds": 60})
            first = await supervisor._idle_tick()
            self.clock += 60
            return first, await supervisor._idle_tick()

        self.assertEqual(asyncio.run(scenario()), (False, True))

    def test_idle_watcher_starts_with_the_service_and_stops_with_it(self) -> None:
        async def scenario():
            await supervisor.startup()
            task = supervisor.lifecycle["watch_task"]
            running = task is not None and not task.done()
            await supervisor.shutdown()
            await asyncio.sleep(0)
            return running, task.cancelled(), supervisor.lifecycle["watch_task"]

        running, cancelled, remaining = asyncio.run(scenario())
        self.assertTrue(running)
        self.assertTrue(cancelled)
        self.assertIsNone(remaining)
        self.assertEqual(supervisor.state["state"], "unloaded")
        self.assertEqual(self.stopped, ["supervisor-shutdown"])

    # ---------------------------------------------------- speech while unloaded

    def test_wait_mode_loads_the_model_and_then_admits_the_request(self) -> None:
        async def scenario():
            payload = {"model": LARGE, "input": "Hello.", "load_wait_s": 5, "_load_wait_s": None}
            return await supervisor._assert_synthesis_ready(payload), payload

        model_id, payload = asyncio.run(scenario())
        self.assertEqual(model_id, LARGE)
        self.assertEqual(self.started, [LARGE])
        self.assertEqual(supervisor.state["state"], "loaded")
        self.assertNotIn("load_wait_s", payload)
        self.assertNotIn("_load_wait_s", payload)
        self.assertEqual(payload["_engine_epoch"], 1)

    def test_a_request_without_a_model_name_loads_the_default_model(self) -> None:
        model_id = asyncio.run(supervisor._assert_synthesis_ready({"input": "Hello."}))
        self.assertEqual(model_id, LARGE)
        self.assertEqual(self.started, [LARGE])

    def test_retry_mode_answers_loading_at_once_and_the_load_carries_on(self) -> None:
        supervisor.lifecycle_settings["speech_when_unloaded"] = "retry"

        async def scenario():
            self.gate = asyncio.Event()
            try:
                await supervisor._assert_synthesis_ready({"model": LARGE})
                raised = None
            except HTTPException as exc:
                raised = exc
            self.gate.set()
            await self._finish_load()
            return raised

        raised = asyncio.run(scenario())
        self.assertEqual(raised.status_code, 503)
        self.assertEqual(raised.detail["state"], "loading")
        self.assertEqual(raised.detail["reason"], "load-in-progress")
        self.assertIs(raised.detail["audio_started"], False)
        self.assertEqual(raised.headers["Retry-After"], str(raised.detail["retryAfterS"]))
        self.assertEqual(raised.headers["X-TTS-Audio-Started"], "false")
        self.assertEqual(supervisor.state["state"], "loaded")  # the refused request did not cancel the load

    def test_refuse_mode_keeps_the_explicit_load_only_answer(self) -> None:
        supervisor.lifecycle_settings["speech_when_unloaded"] = "refuse"
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(supervisor._assert_synthesis_ready({"model": LARGE, "load_wait_s": 30}))
        self.assertEqual((raised.exception.status_code, raised.exception.detail), (409, LEGACY_NOT_LOADED))
        self.assertEqual(self.started, [])

    def test_a_request_can_ask_for_an_immediate_answer_or_a_longer_wait(self) -> None:
        async def refused(payload):
            self.gate = asyncio.Event()
            try:
                await supervisor._assert_synthesis_ready(payload)
                raised = None
            except HTTPException as exc:
                raised = exc
            self.gate.set()
            await self._finish_load()
            return raised

        body = asyncio.run(refused({"model": LARGE, "load_wait_s": 0}))
        self.assertEqual((body.status_code, body.detail["state"]), (503, "loading"))
        self.tearDown()
        self.setUp()
        header = asyncio.run(refused({"model": LARGE, "_load_wait_s": "0"}))
        self.assertEqual((header.status_code, header.detail["state"]), (503, "loading"))
        self.tearDown()
        self.setUp()
        supervisor.lifecycle_settings["speech_when_unloaded"] = "retry"
        waited = asyncio.run(supervisor._assert_synthesis_ready({"model": LARGE, "load_wait_s": 5}))
        self.assertEqual(waited, LARGE)
        for bad in ("soon", -1, 601, True):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(supervisor._assert_synthesis_ready({"model": LARGE, "load_wait_s": bad}))
            self.assertEqual(raised.exception.status_code, 422)

    def test_a_cancelled_request_does_not_cancel_the_load(self) -> None:
        async def scenario():
            self.gate = asyncio.Event()
            request = asyncio.create_task(supervisor._assert_synthesis_ready({"model": LARGE}))
            await asyncio.sleep(0.05)
            request.cancel()
            try:
                await request
            except asyncio.CancelledError:
                pass
            load = supervisor.lifecycle["load_task"]
            still_loading = load is not None and not load.done()
            self.gate.set()
            await load
            return still_loading

        self.assertTrue(asyncio.run(scenario()))
        self.assertEqual(supervisor.state["state"], "loaded")
        self.assertEqual(self.started, [LARGE])

    def test_a_guard_refusal_reaches_the_speech_request_as_not_spoken(self) -> None:
        self.guard = {"ok": False, "reason": "GPU busy: 97% utilization.", "utilizationPercent": 97}
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(supervisor._assert_synthesis_ready({"model": LARGE}))
        error = raised.exception
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.detail["state"], "blocked")
        self.assertIn("97% utilization", error.detail["reason"])
        self.assertIs(error.detail["audio_started"], False)
        self.assertEqual(error.headers["X-TTS-Audio-Started"], "false")
        self.assertEqual(self.started, [])

    def test_a_failed_load_reaches_the_waiting_request_and_is_not_left_loading(self) -> None:
        self.fail_start = True
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(supervisor._assert_synthesis_ready({"model": LARGE}))
        error = raised.exception
        self.assertEqual((error.status_code, error.detail["state"]), (503, "error"))
        self.assertIn("engine binary is missing", error.detail["message"])
        self.assertIs(error.detail["audio_started"], False)
        self.assertEqual(supervisor.state["state"], "error")
        self.assertIn("engine binary is missing", supervisor._lifecycle_status()["lastLoadError"])

    def test_a_failed_load_is_not_retried_on_every_poll(self) -> None:
        async def scenario():
            self.fail_start = True
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            self.fail_start = False
            soon = await supervisor.ensure_model(LARGE)
            started_soon = list(self.started)
            self.clock += supervisor.LOAD_RETRY_COOLDOWN_SECONDS + 1
            later = await supervisor.ensure_model(LARGE)
            await self._finish_load()
            return soon, started_soon, later

        soon, started_soon, later = asyncio.run(scenario())
        self.assertEqual((soon["accepted"], soon["reason"], soon["retryable"]), (False, "load-failed", True))
        self.assertIn("engine binary is missing", soon["message"])
        self.assertEqual(soon["state"], "error")
        self.assertEqual(started_soon, [])
        self.assertEqual((later["accepted"], later["reason"]), (True, "load-in-progress"))
        self.assertEqual(supervisor.state["state"], "loaded")

    def test_a_manual_load_is_not_held_back_by_a_failed_load(self) -> None:
        async def scenario():
            self.fail_start = True
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            self.fail_start = False
            answer = await supervisor.backend_switch({"model_key": LARGE, "wait": False})
            await self._finish_load()
            return answer

        answer = asyncio.run(scenario())
        self.assertEqual((answer["accepted"], answer["reason"]), (True, "load-in-progress"))
        self.assertEqual(supervisor.state["state"], "loaded")

    def test_speech_never_switches_the_resident_model(self) -> None:
        async def scenario():
            await supervisor.ensure_model(LARGE)
            await self._finish_load()
            self.clock += 10000
            await supervisor._assert_synthesis_ready({"model": SMALL})

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(scenario())
        self.assertEqual((raised.exception.status_code, raised.exception.detail), (409, LEGACY_NOT_ACTIVE))
        self.assertEqual(self.started, [LARGE])

    def test_an_unknown_model_name_keeps_todays_answer(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(supervisor._assert_synthesis_ready({"model": "tts-1"}))
        self.assertEqual((raised.exception.status_code, raised.exception.detail), (409, LEGACY_NOT_ACTIVE))
        self.assertEqual(self.started, [])

    def test_load_wait_header_is_read_when_a_request_has_headers(self) -> None:
        class Request:
            headers = {"x-tts-load-wait-seconds": "0"}

        self.assertEqual(supervisor._load_wait_header(Request()), "0")
        self.assertIsNone(supervisor._load_wait_header(object()))

    # ---------------------------------------------------------- settings/status

    def test_settings_are_validated_and_persisted(self) -> None:
        async def scenario():
            return await supervisor.control_lifecycle_update({
                "keepLoaded": True, "idleUnloadSeconds": 300, "speechWhenUnloaded": "retry",
                "loadWaitSeconds": 30, "wakeOnHealth": True})

        status = asyncio.run(scenario())["lifecycle"]
        self.assertEqual((status["keepLoaded"], status["idleUnloadSeconds"], status["speechWhenUnloaded"],
                          status["loadWaitSeconds"], status["wakeOnHealth"]), (True, 300, "retry", 30, True))
        saved = json.loads(supervisor.LIFECYCLE_SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved["idle_unload_seconds"], 300)
        self.assertEqual(supervisor._load_lifecycle_settings(), dict(supervisor.lifecycle_settings))
        self.assertEqual(supervisor.EVENTS[-1]["action"], "lifecycle-updated")
        for bad in ({"idleUnloadSeconds": 10}, {"idleUnloadSeconds": 1.5}, {"speechWhenUnloaded": "maybe"},
                    {"keepLoaded": "yes"}, {"loadWaitSeconds": 601}, {"idleUnloadSeconds": True}):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(supervisor.control_lifecycle_update(bad))
            self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(supervisor.lifecycle_settings["idle_unload_seconds"], 300)  # a bad update changed nothing
        supervisor.LIFECYCLE_SETTINGS_PATH.write_text("{not json", encoding="utf-8")
        self.assertEqual(supervisor._load_lifecycle_settings(), supervisor._default_lifecycle_settings())
        supervisor.LIFECYCLE_SETTINGS_PATH.unlink()

    def test_status_reports_the_lifecycle_block(self) -> None:
        async def scenario():
            self.gate = asyncio.Event()
            await supervisor.ensure_model(LARGE)
            await asyncio.sleep(0)
            self.clock += 4
            loading = (await supervisor.control_status())["lifecycle"]
            self.gate.set()
            await self._finish_load()
            self.clock += 30
            loaded = (await supervisor.control_status())["lifecycle"]
            listed = (await supervisor.backend_models())["lifecycle"]
            only = (await supervisor.control_lifecycle())["lifecycle"]
            return loading, loaded, listed, only

        loading, loaded, listed, only = asyncio.run(scenario())
        self.assertEqual(set(loaded), {"keepLoaded", "idleUnloadSeconds", "idleForS", "unloadInS",
                                       "speechWhenUnloaded", "loadWaitSeconds", "wakeOnHealth", "manualHold",
                                       "busy", "load", "lastUnloadReason", "lastLoadError"})
        self.assertEqual(loading["load"]["model"], LARGE)
        self.assertEqual(loading["load"]["phase"], "loading")
        self.assertEqual(loading["load"]["elapsedS"], 4.0)
        self.assertEqual(loading["load"]["etaS"], 56)
        self.assertRegex(loading["load"]["startedAt"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertIsNone(loading["idleForS"])
        self.assertIsNone(loaded["load"])
        self.assertEqual((loaded["idleForS"], loaded["unloadInS"]), (30.0, 870.0))
        self.assertEqual(listed, loaded)
        self.assertEqual(only, loaded)

    def test_wake_on_health_is_off_by_default_and_loads_when_enabled(self) -> None:
        async def scenario():
            await supervisor.health()
            quiet = list(self.started), supervisor.lifecycle["load_task"]
            supervisor.lifecycle_settings["wake_on_health"] = True
            await supervisor.health()
            await self._finish_load()
            return quiet

        quiet = asyncio.run(scenario())
        self.assertEqual(quiet, ([], None))
        self.assertEqual(self.started, [LARGE])
        self.assertEqual(supervisor.state["state"], "loaded")

    def test_disabled_guard_skips_the_gpu_query_for_speech_but_not_for_loads(self) -> None:
        calls = []

        def query(*_args, **_kwargs):
            calls.append(1)
            return "12000, 10\n"

        saved = dict(supervisor.gpu_guard_settings)
        try:
            with patch.object(supervisor.subprocess, "check_output", query):
                supervisor.gpu_guard_settings["mode"] = "disabled"
                speech = REAL_GPU_GUARD(LARGE, operation="synthesis")
                self.assertEqual(calls, [])
                load = REAL_GPU_GUARD(LARGE, operation="load")
                self.assertEqual(calls, [1])
                supervisor.gpu_guard_settings["mode"] = "enforced"
                enforced = REAL_GPU_GUARD(LARGE, operation="synthesis")
                self.assertEqual(calls, [1, 1])
        finally:
            supervisor.gpu_guard_settings.clear()
            supervisor.gpu_guard_settings.update(saved)
        self.assertEqual((speech["ok"], speech["bypassed"], speech["guardMode"]), (True, True, "disabled"))
        self.assertEqual((load["ok"], load["bypassed"], load["freeMiB"]), (True, True, 12000))
        self.assertEqual((enforced["ok"], enforced["freeMiB"]), (True, 12000))

    def test_blocking_calls_run_off_the_event_loop(self) -> None:
        source = SUPERVISOR.read_text(encoding="utf-8")
        self.assertIn("await asyncio.to_thread(_stop_engine, action=\"old-model-released\")", source)
        self.assertIn("return await asyncio.to_thread(gpu_guard, model_id, **options)", source)
        self.assertIn("guard = await gpu_guard_async(model_id, operation=\"synthesis\")", source)
        self.assertIn("lifecycle[\"watch_task\"] = asyncio.create_task(_idle_watch())", source)
        # Background work is created only inside startup(), which a gateway may
        # call itself; nothing starts when the module is imported.
        self.assertIsNone(_load_supervisor().lifecycle["watch_task"])


class LifecycleHttpTests(_LifecycleFixture):
    """The same stub engine, driven through the real routes and JSON."""

    def _client(self):
        from fastapi.testclient import TestClient

        return TestClient(supervisor.app)

    def test_ensure_status_and_settings_over_http(self) -> None:
        import time as real_time

        with self._client() as client:
            first = client.post("/v1/backend/models/ensure")
            self.assertEqual(first.status_code, 200)
            self.assertEqual((first.json()["accepted"], first.json()["model"]), (True, LARGE))
            deadline = real_time.monotonic() + 10
            answer = first.json()
            while not answer["ready"] and real_time.monotonic() < deadline:
                real_time.sleep(0.05)
                answer = client.post("/v1/backend/models/ensure", json={"model": LARGE}).json()
            self.assertTrue(answer["ready"])
            self.assertEqual(self.started, [LARGE])

            status = client.get("/control/status").json()
            self.assertEqual(status["state"], "loaded")
            self.assertEqual(status["lifecycle"]["idleUnloadSeconds"], 900)
            self.assertIs(status["singleResident"], True)

            saved = client.put("/control/lifecycle", json={"keepLoaded": True, "idleUnloadSeconds": 120})
            self.assertEqual(saved.status_code, 200)
            self.assertEqual((saved.json()["lifecycle"]["keepLoaded"], saved.json()["lifecycle"]["idleUnloadSeconds"]),
                             (True, 120))
            posted = client.post("/control/lifecycle", json={"keepLoaded": False})
            self.assertEqual(posted.json()["lifecycle"]["keepLoaded"], False)
            self.assertEqual(client.get("/control/lifecycle").json()["lifecycle"]["idleUnloadSeconds"], 120)
            bad = client.put("/control/lifecycle", json={"idleUnloadSeconds": 5})
            self.assertEqual(bad.status_code, 422)
            self.assertIn("between 60 and 86400", bad.json()["detail"])

            unloaded = client.post("/v1/backend/models/unload")
            self.assertEqual(unloaded.json()["state"], "unloaded")
            held = client.post("/v1/backend/models/ensure", json={"model": LARGE}).json()
            self.assertEqual((held["accepted"], held["reason"]), (False, "manual-hold"))
            self.assertEqual(client.post("/v1/backend/models/ensure", json={"model": "nope"}).status_code, 404)
        supervisor.LIFECYCLE_SETTINGS_PATH.unlink(missing_ok=True)

    def test_speech_while_unloaded_answers_retry_after_over_http(self) -> None:
        supervisor.lifecycle_settings["speech_when_unloaded"] = "retry"
        self.gate = None
        with self._client() as client:
            self.gate = asyncio.Event()
            response = client.post("/v1/audio/speech", json={"model": LARGE, "input": "Hello.", "response_format": "pcm"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["x-tts-audio-started"], "false")
            self.assertGreaterEqual(int(response.headers["retry-after"]), 1)
            detail = response.json()["detail"]
            self.assertEqual((detail["state"], detail["reason"], detail["audio_started"]),
                             ("loading", "load-in-progress", False))
            header = client.post("/v1/audio/speech", headers={"X-TTS-Load-Wait-Seconds": "0"},
                                 json={"model": LARGE, "input": "Hello.", "response_format": "pcm"})
            self.assertEqual(header.status_code, 503)
            invalid = client.post("/v1/audio/speech", headers={"X-TTS-Load-Wait-Seconds": "soon"},
                                  json={"model": LARGE, "input": "Hello.", "response_format": "pcm"})
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(self.started, [LARGE])  # one load, however many requests asked


class VoiceStudioLifecycleContractTests(unittest.TestCase):
    """The Studio exposes the lifecycle settings and loads on demand when it generates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ui = (INTEGRATION / "gradio_voice_studio.py").read_text(encoding="utf-8")

    def test_studio_has_the_keep_loaded_switch_and_quiet_time(self) -> None:
        for expected in (
            'gr.Accordion("Model loading and idle unload", open=False)',
            'label="Keep the model loaded"',
            'label="Unload after this many minutes without speech"',
            'label="When speech arrives and no model is loaded"',
            '["wait", "retry", "refuse"]',
            'gr.Button("Apply loading settings", variant="secondary")',
        ):
            self.assertIn(expected, self.ui)
        # Manual controls stay.
        self.assertIn('gr.Button("Load selected model", variant="secondary")', self.ui)
        self.assertIn('gr.Button("Unload model", variant="stop")', self.ui)

    def test_studio_reads_and_saves_settings_through_the_lifecycle_route(self) -> None:
        self.assertIn('httpx.get(f"{base_url.rstrip(\'/\')}/control/lifecycle", timeout=10.0)', self.ui)
        self.assertIn('httpx.put(f"{base_url.rstrip(\'/\')}/control/lifecycle", json=payload, timeout=10.0)', self.ui)
        self.assertIn('"idleUnloadSeconds": int(idle_minutes) * 60,', self.ui)
        self.assertIn("fn=load_lifecycle_settings,", self.ui)
        self.assertIn("fn=save_lifecycle_settings,", self.ui)

    def test_studio_generation_loads_on_demand_and_respects_a_hold(self) -> None:
        self.assertIn('ensure_url = f"{base_url.rstrip(\'/\')}/v1/backend/models/ensure"', self.ui)
        self.assertIn('json={"model": model_key, "switch": "never"}', self.ui)
        self.assertIn(
            'return bool(info) and info.get("speechWhenUnloaded") != "refuse" and not info.get("manualHold")',
            self.ui,
        )
        self.assertIn('if not loaded and can_load_on_demand(data.get("lifecycle") or {}):', self.ui)
        # The explicit-load message remains for the refuse policy and a manual hold.
        self.assertIn("Click Load selected model first.", self.ui)

    def test_studio_payload_is_accepted_by_the_supervisor(self) -> None:
        payload = {"keepLoaded": True, "idleUnloadSeconds": 20 * 60, "loadWaitSeconds": 45,
                   "speechWhenUnloaded": "retry", "wakeOnHealth": False}
        updated = supervisor._validated_lifecycle_settings(payload, supervisor._default_lifecycle_settings())
        self.assertEqual(updated, {"keep_loaded": True, "idle_unload_seconds": 1200, "load_wait_seconds": 45,
                                   "speech_when_unloaded": "retry", "wake_on_health": False})


if __name__ == "__main__":
    unittest.main()
