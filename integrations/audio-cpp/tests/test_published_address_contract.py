"""The bare address of each published port leads to Voice Studio.

Docker Desktop and similar tools link a published port to its bare address
(`http://localhost:8891/`, `http://localhost:8890/`). These checks pin where
those two addresses lead, and that the Realtime test page keeps an address.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).parents[3]
INTEGRATION = Path(__file__).parents[1]
WEB = ROOT / "web" / "hf-realtime-voice"
COMPOSE = INTEGRATION / "compose.candidate.yml"

gradio_stub = types.ModuleType("gradio")
gradio_stub.Blocks = object
gradio_stub.themes = types.SimpleNamespace(Base=object)
sys.modules.setdefault("gradio", gradio_stub)

_scratch = tempfile.mkdtemp(prefix="published-address-")
with patch.dict(os.environ, {"VOICE_LIBRARY_DIR": _scratch, "LOAD_BALANCER_URL": "", "SPACE_ID": ""}):
    sys.path.insert(0, str(WEB))
    try:
        server_spec = importlib.util.spec_from_file_location("published_address_web_host_test", WEB / "server.py")
        assert server_spec and server_spec.loader
        server = importlib.util.module_from_spec(server_spec)
        server_spec.loader.exec_module(server)
    finally:
        sys.path.remove(str(WEB))

    supervisor_spec = importlib.util.spec_from_file_location(
        "published_address_supervisor_test", INTEGRATION / "supervisor.py"
    )
    assert supervisor_spec and supervisor_spec.loader
    supervisor = importlib.util.module_from_spec(supervisor_spec)
    supervisor_spec.loader.exec_module(supervisor)

STUDIO_URL = "http://127.0.0.1:8891/voice-studio/"


class StudioPortTest(unittest.TestCase):
    """The web host, published as 8891."""

    def setUp(self) -> None:
        # No lifespan: the routes under test need no startup work.
        self.client = TestClient(server.app, follow_redirects=False)

    def test_bare_address_opens_voice_studio_where_it_is_mounted(self) -> None:
        with patch.object(server, "VOICE_STUDIO_MOUNTED", True), patch.dict(os.environ, {"AUDIO_CPP_ROOT_PAGE": ""}):
            response = self.client.get("/")
            self.assertEqual(response.status_code, 307)
            self.assertEqual(response.headers["location"], "/voice-studio/")
            # Gradio reads options such as the theme from the query string.
            self.assertEqual(self.client.get("/?__theme=dark").headers["location"], "/voice-studio/?__theme=dark")
            self.assertEqual(self.client.head("/").status_code, 307)

    def test_voice_studio_without_the_trailing_slash_is_not_a_dead_end(self) -> None:
        with patch.object(server, "VOICE_STUDIO_MOUNTED", True):
            response = self.client.get("/voice-studio")
            self.assertEqual(response.status_code, 307)
            self.assertEqual(response.headers["location"], "/voice-studio/")
        with patch.object(server, "VOICE_STUDIO_MOUNTED", False):
            self.assertEqual(self.client.get("/voice-studio").status_code, 404)

    def test_realtime_page_keeps_an_address_and_can_stay_at_the_root(self) -> None:
        with patch.object(server, "VOICE_STUDIO_MOUNTED", True):
            for path in ("/realtime", "/realtime/"):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 307)
                self.assertEqual(response.headers["location"], "/index.html")
            page = self.client.get("/index.html")
            self.assertEqual(page.status_code, 200)
            self.assertIn("main.js", page.text)
            with patch.dict(os.environ, {"AUDIO_CPP_ROOT_PAGE": "realtime"}):
                root = self.client.get("/")
                self.assertEqual(root.status_code, 200)
                self.assertEqual(root.text, page.text)

    def test_without_a_mounted_voice_studio_the_root_is_the_realtime_page(self) -> None:
        # The plain Realtime server (no Gradio) must behave as it always did.
        with patch.object(server, "VOICE_STUDIO_MOUNTED", False), patch.dict(os.environ, {"AUDIO_CPP_ROOT_PAGE": ""}):
            root = self.client.get("/")
            self.assertEqual(root.status_code, 200)
            self.assertIn("<!doctype html>", root.text.lower())
            self.assertEqual(server._root_page(), "realtime")
            with patch.dict(os.environ, {"AUDIO_CPP_ROOT_PAGE": "voice-studio"}):
                self.assertEqual(server._root_page(), "realtime")

    def test_api_routes_and_static_files_are_untouched(self) -> None:
        with patch.object(server, "VOICE_STUDIO_MOUNTED", True):
            self.assertEqual(self.client.get("/main.js").status_code, 200)
            self.assertEqual(self.client.get("/api/ui-settings").status_code, 200)

    def test_start_up_names_the_published_addresses(self) -> None:
        env = {"AUDIO_CPP_STUDIO_PUBLIC_URL": STUDIO_URL, "AUDIO_CPP_API_PUBLIC_URL": "http://127.0.0.1:8890/"}
        with patch.object(server, "VOICE_STUDIO_MOUNTED", True), patch.dict(os.environ, env):
            self.assertEqual(
                server._published_addresses(),
                [f"Voice Studio: {STUDIO_URL}", "Speech API:   http://127.0.0.1:8890/v1"],
            )
        with patch.object(server, "VOICE_STUDIO_MOUNTED", True), patch.dict(
            os.environ, {"AUDIO_CPP_STUDIO_PUBLIC_URL": "", "AUDIO_CPP_API_PUBLIC_URL": ""}
        ):
            lines = server._published_addresses()
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].endswith("/voice-studio/"))
        with patch.object(server, "VOICE_STUDIO_MOUNTED", False), patch.dict(
            os.environ, {"AUDIO_CPP_STUDIO_PUBLIC_URL": STUDIO_URL, "AUDIO_CPP_API_PUBLIC_URL": ""}
        ):
            self.assertEqual(server._published_addresses(), [])


class ApiPortTest(unittest.TestCase):
    """The supervisor, published as 8890."""

    def setUp(self) -> None:
        # No lifespan: the supervisor's startup and shutdown hooks manage the engine.
        self.client = TestClient(supervisor.app, follow_redirects=False)

    def test_a_browser_is_sent_to_voice_studio(self) -> None:
        with patch.dict(os.environ, {"AUDIO_CPP_STUDIO_PUBLIC_URL": STUDIO_URL}):
            response = self.client.get("/", headers={"accept": "text/html,application/xhtml+xml,*/*;q=0.8"})
            self.assertEqual(response.status_code, 307)
            self.assertEqual(response.headers["location"], STUDIO_URL)

    def test_an_api_client_is_told_where_things_are(self) -> None:
        with patch.dict(os.environ, {"AUDIO_CPP_STUDIO_PUBLIC_URL": STUDIO_URL}):
            response = self.client.get("/", headers={"accept": "application/json"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "service": "audio.cpp Qwen3-TTS speech server",
                    "voiceStudioUrl": STUDIO_URL,
                    "apiBasePath": "/v1",
                    "healthPath": "/health",
                    "statusPath": "/control/status",
                },
            )

    def test_without_a_usable_address_nobody_is_redirected(self) -> None:
        for value in ("", "javascript:alert(1)", "//elsewhere.example/"):
            with patch.dict(os.environ, {"AUDIO_CPP_STUDIO_PUBLIC_URL": value}):
                response = self.client.get("/", headers={"accept": "text/html"})
                self.assertEqual(response.status_code, 200, value)
                self.assertIsNone(response.json()["voiceStudioUrl"], value)


class ComposeTest(unittest.TestCase):
    def test_compose_passes_the_addresses_it_publishes(self) -> None:
        compose = COMPOSE.read_text(encoding="utf-8")
        # The same variables set the published ports and the reported addresses,
        # so an instance on other ports (side by side) reports its own.
        self.assertIn('"127.0.0.1:${AUDIO_CPP_STUDIO_PORT:-8891}:7860"', compose)
        self.assertIn(
            "AUDIO_CPP_STUDIO_PUBLIC_URL: http://127.0.0.1:${AUDIO_CPP_STUDIO_PORT:-8891}/voice-studio/", compose
        )
        self.assertIn('"127.0.0.1:${AUDIO_CPP_API_PORT:-8890}:8080"', compose)
        self.assertIn("AUDIO_CPP_API_PUBLIC_URL: http://127.0.0.1:${AUDIO_CPP_API_PORT:-8890}", compose)


if __name__ == "__main__":
    unittest.main()
