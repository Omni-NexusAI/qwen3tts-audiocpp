"""Voice steadiness and latency benchmark for the Qwen3-TTS audio.cpp speech server.

The tool only reads server state and sends speech requests.  It never loads,
unloads, switches or saves anything on the server, and it refuses to run when
no model is loaded.

It speaks a fixed set of multi-sentence replies in several request shapes
("arms") with one fixed seed per reply, keeps the audio, and reports:

* time to first audio and synthesis time per second of audio;
* how much the voice moves inside one reply: the spread of median pitch and of
  spectral centroid (brightness) between the requests of a reply and between
  equal segments of the whole reply's audio;
* for the same text and seed, the sample-level difference between streamed and
  whole-phrase delivery, and whether it concentrates at streaming block edges;
* the engine's own stage timings, including a fit of talker time against
  generated frames (the fixed cost before the first frame).

Usage:

    python integrations/audio-cpp/tools/voice_bench.py arms
    python integrations/audio-cpp/tools/voice_bench.py run --arms sentences-stream,sentences-whole
    python integrations/audio-cpp/tools/voice_bench.py report

Audio and results are written under ``.runtime/voice-bench`` by default, which
is outside version control.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROVIDER = "qwen3tts-audiocpp"
SAMPLE_RATE = 24000
SAMPLES_PER_CODEC_FRAME = 1920  # 24 kHz output at 12.5 codec frames per second
DEFAULT_OUT = Path(".runtime") / "voice-bench"
DEFAULT_BASE_URL = "http://127.0.0.1:8890"

# Ordinary assistant replies, five sentences each so every reply has the same
# number of requests in the sentence arms.  Plain text without abbreviations,
# digits or markup, and no sentence shorter than six words.
REPLIES = [
    [
        "Frankfort is the capital of Kentucky, and it sits on the Kentucky River.",
        "The city is small, with roughly twenty-eight thousand residents.",
        "Its capitol building was completed in nineteen ten and is open to visitors.",
        "The old state capitol nearby is now a museum of Kentucky history.",
        "I can also look up the weather there if you would like.",
    ],
    [
        "Your calendar is clear until two this afternoon, so the morning is yours.",
        "After that you have the design review, which is scheduled for an hour.",
        "The team sent the slides last night, and I have saved them to your desktop.",
        "There is also a reminder to call the dentist before the office closes.",
        "Tell me if you want any of those moved to another day.",
    ],
    [
        "The recipe needs two cups of flour, one egg and a little melted butter.",
        "Mix the dry ingredients first, then stir in the milk a bit at a time.",
        "Let the batter rest for ten minutes while the pan warms over medium heat.",
        "Each pancake takes about two minutes per side, until the edges look dry.",
        "Serve them warm, with fruit or syrup, whichever you happen to prefer.",
    ],
    [
        "The download finished a moment ago and the file passed its integrity check.",
        "I unpacked it into the projects folder and left the archive where it was.",
        "Two of the scripts needed execute permission, which I have already granted.",
        "The test suite ran cleanly, apart from one test that is marked as skipped.",
        "Everything is ready whenever you want to start the build.",
    ],
    [
        "The train to the coast leaves at a quarter past nine from platform four.",
        "It reaches the harbour station a little before noon, with one short stop.",
        "A return ticket costs less if you travel back after seven in the evening.",
        "The forecast there is dry and bright, though the wind will be fairly strong.",
        "Shall I reserve a seat by the window for the outward journey?",
    ],
    [
        "The printer has finished the first layer and the adhesion looks even.",
        "Nozzle temperature is holding steady, and the bed is two degrees above target.",
        "At the current speed the whole part should take just under three hours.",
        "I will keep watching the camera and pause the job if anything lifts.",
        "You do not need to stay nearby unless you would like to.",
    ],
    [
        "That word comes from an old French term meaning a small enclosed garden.",
        "It entered English in the fourteenth century, along with many cooking words.",
        "The spelling changed several times before printers settled on the modern form.",
        "Today it is used far more often in writing than in everyday speech.",
        "There is a related verb, although hardly anyone uses it any more.",
    ],
    [
        "The living room lights are now at forty percent and set to a warm tone.",
        "I closed the blinds on the west side because the sun was on the screen.",
        "The thermostat will hold twenty-one degrees until you go to bed tonight.",
        "The front door is locked, and the garage door closed ten minutes ago.",
        "Let me know if you want the music turned down as well.",
    ],
]

# Very short phrases: a streamed request must speak these with no fallback.
SHORT_PHRASES = [
    "Yes.",
    "No.",
    "Certainly.",
    "Done.",
    "Right away.",
    "One moment.",
    "Good evening.",
    "Of course, sir.",
    "Not yet.",
    "Thank you.",
]

# Reply i uses seed i in every arm, so arms are compared on identical sampling.
SEEDS = [314159, 271828, 161803, 141421, 173205, 223606, 244948, 264575]

WARMUP_TEXT = "This first request only warms the engine and is not measured."

# split: "sentences" sends one request per sentence, "reply" sends the whole
# reply as one request, "short" sends each short phrase as its own reply.
# delivery: "stream" is pcm + stream:true, "whole" is pcm + stream:false.
ARMS: dict[str, dict[str, Any]] = {
    "sentences-stream": {
        "split": "sentences", "delivery": "stream",
        "about": "One streamed request per sentence, the shape most agent frameworks send.",
    },
    "sentences-whole": {
        "split": "sentences", "delivery": "whole",
        "about": "One whole-phrase PCM request per sentence; pairs with sentences-stream.",
    },
    "one-request-stream": {
        "split": "reply", "delivery": "stream",
        "about": "The whole reply as one streamed request: the steadiness floor.",
    },
    "one-request-whole": {
        "split": "reply", "delivery": "whole",
        "about": "The whole reply as one whole-phrase request; pairs with one-request-stream.",
    },
    "emulated-continuation": {
        "split": "sentences", "delivery": "whole", "continuation": "emulated",
        "about": "Per sentence, with the reference extended by the reply's earlier audio and text. "
                 "Approximates continuation with today's server; slower, and not exact.",
    },
    "continuation": {
        "split": "sentences", "delivery": "stream", "continuation": "id",
        "about": "Per sentence with a continuation_id per reply. Needs a server that supports it.",
    },
    "short-stream": {
        "split": "short", "delivery": "stream",
        "about": "Very short phrases, streamed.",
    },
    "short-whole": {
        "split": "short", "delivery": "whole",
        "about": "Very short phrases, whole-phrase PCM.",
    },
}
DEFAULT_ARMS = ["sentences-stream", "sentences-whole", "one-request-stream"]
PAIRS = [("sentences-stream", "sentences-whole"), ("one-request-stream", "one-request-whole"),
         ("short-stream", "short-whole")]


# --------------------------------------------------------------------- analysis

def frame_pitch(frame: np.ndarray, rate: int, *, gate: float = 300.0, low: float = 70.0,
                high: float = 400.0, threshold: float = 0.45) -> float | None:
    """Autocorrelation pitch of one frame in Hz, or None when it is not voiced."""
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame ** 2)) < gate:
        return None
    first, last = int(rate / high), int(rate / low)
    correlation = np.correlate(frame, frame, "full")[len(frame) - 1:]
    if correlation[0] <= 0 or last >= len(correlation):
        return None
    segment = correlation[first:last] / correlation[0]
    lag = int(np.argmax(segment)) + first
    return rate / lag if segment[lag - first] > threshold else None


def features(pcm: bytes, rate: int, *, octave_guard: float = 0.6) -> tuple[float | None, float | None]:
    """Median pitch and median spectral centroid over the voiced 40 ms frames.

    Frames whose pitch lies more than ``octave_guard`` octaves from the clip's
    median are dropped first: a raw autocorrelation peak sometimes lands an
    octave away, which would otherwise read as a pitch jump.
    """
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float64)
    size = int(rate * 0.04)
    if size <= 0 or len(samples) < size:
        return None, None
    window = np.hanning(size)
    frequencies = np.fft.rfftfreq(size, 1 / rate)
    pitches: list[float] = []
    centroids: list[float] = []
    for start in range(0, len(samples) - size + 1, size):
        frame = samples[start:start + size]
        pitch = frame_pitch(frame, rate)
        if pitch is None:
            continue
        spectrum = np.abs(np.fft.rfft(frame * window))
        pitches.append(pitch)
        centroids.append(float((spectrum * frequencies).sum() / max(spectrum.sum(), 1e-9)))
    if not pitches:
        return None, None
    if octave_guard:
        centre = statistics.median(pitches)
        kept = [index for index, value in enumerate(pitches) if abs(math.log2(value / centre)) <= octave_guard]
        if kept:
            pitches = [pitches[index] for index in kept]
            centroids = [centroids[index] for index in kept]
    return statistics.median(pitches), statistics.median(centroids)


def segment_features(pcm: bytes, rate: int, parts: int = 3) -> list[dict[str, float | None]]:
    """Pitch and brightness of each equal segment of one clip."""
    size = (len(pcm) // 2 // parts) * 2
    found = []
    for number in range(parts):
        pitch, centroid = features(pcm[number * size:(number + 1) * size], rate) if size else (None, None)
        found.append({"pitch_hz": rounded(pitch), "centroid_hz": rounded(centroid)})
    return found


def rounded(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)


def spread(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    """(max minus min, standard deviation) of the present values; needs two."""
    present = [value for value in values if value is not None]
    if len(present) < 2:
        return None, None
    return max(present) - min(present), statistics.pstdev(present)


def median(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def percentile(values: Iterable[float | None], fraction: float) -> float | None:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    position = fraction * (len(present) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(present) - 1)
    return present[lower] + (present[upper] - present[lower]) * (position - lower)


def block_boundaries(first_frames: int, steady_frames: int, total_samples: int) -> list[int]:
    """Sample offsets where one streamed block ends and the next begins."""
    if first_frames <= 0 or steady_frames <= 0:
        return []
    boundaries = []
    offset = first_frames * SAMPLES_PER_CODEC_FRAME
    while offset < total_samples:
        boundaries.append(offset)
        offset += steady_frames * SAMPLES_PER_CODEC_FRAME
    return boundaries


def pcm_difference(first: bytes, second: bytes, rate: int, boundaries: Iterable[int] = (),
                   *, edge_ms: float = 10.0) -> dict[str, Any] | None:
    """Sample-level difference between two clips of the same text and seed.

    ``edge_ratio`` is the difference's RMS within ``edge_ms`` of a streaming
    block boundary divided by its RMS elsewhere.  A value near one means the
    difference is spread evenly; a large value means it sits at block edges.
    """
    left = np.frombuffer(first[: len(first) - len(first) % 2], dtype="<i2").astype(np.float64)
    right = np.frombuffer(second[: len(second) - len(second) % 2], dtype="<i2").astype(np.float64)
    count = min(len(left), len(right))
    if count == 0:
        return None
    difference = left[:count] - right[:count]
    signal_rms = float(np.sqrt(np.mean(right[:count] ** 2)))
    difference_rms = float(np.sqrt(np.mean(difference ** 2)))
    near = np.zeros(count, dtype=bool)
    reach = max(1, int(rate * edge_ms / 1000))
    for boundary in boundaries:
        if 0 < boundary < count:
            near[max(0, boundary - reach):min(count, boundary + reach)] = True
    edge_rms = float(np.sqrt(np.mean(difference[near] ** 2))) if near.any() else None
    elsewhere_rms = float(np.sqrt(np.mean(difference[~near] ** 2))) if (~near).any() else None
    ratio = edge_rms / elsewhere_rms if edge_rms is not None and elsewhere_rms else None
    return {
        "samples": count,
        "length_delta_samples": len(left) - len(right),
        "identical": not difference.any() and len(left) == len(right),
        "difference_rms": rounded(difference_rms, 2),
        "difference_db_below_signal": rounded(20 * math.log10(signal_rms / difference_rms), 1)
        if difference_rms > 0 and signal_rms > 0 else None,
        "max_abs_difference": int(np.max(np.abs(difference))),
        "edge_rms": rounded(edge_rms, 2),
        "elsewhere_rms": rounded(elsewhere_rms, 2),
        "edge_ratio": rounded(ratio, 2),
    }


def linear_fit(xs: Iterable[float], ys: Iterable[float]) -> tuple[float, float] | None:
    """(intercept, slope) of ys against xs, or None with fewer than three distinct xs."""
    points = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len({x for x, _ in points}) < 3:
        return None
    slope, intercept = np.polyfit([x for x, _ in points], [y for _, y in points], 1)
    return float(intercept), float(slope)


# ------------------------------------------------------------------------ audio

def wav_bytes(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


def read_wav_mono16(data: bytes) -> tuple[int, np.ndarray]:
    """(sample rate, mono int16 samples) of a 16-bit PCM WAV; stereo is averaged."""
    with wave.open(io.BytesIO(data), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError("Only 16-bit PCM WAV references are supported by the benchmark.")
        rate, channels = reader.getframerate(), reader.getnchannels()
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    if channels > 1:
        samples = samples[: len(samples) - len(samples) % channels].reshape(-1, channels).mean(axis=1)
    return rate, np.asarray(samples, dtype=np.float64)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(samples) == 0:
        return np.asarray(samples, dtype=np.float64)
    length = int(round(len(samples) * target_rate / source_rate))
    positions = np.linspace(0, len(samples) - 1, num=length)
    return np.interp(positions, np.arange(len(samples)), samples)


def continuation_reference(reference_wav: bytes, reference_text: str, previous: list[tuple[str, bytes]], *,
                           budget_seconds: float = 12.0, gap_ms: int = 150,
                           rate: int = SAMPLE_RATE) -> tuple[bytes, str, float]:
    """Extend a clone reference with the reply's earlier phrases.

    Returns (WAV bytes, transcript, seconds of earlier audio used).  The most
    recent whole phrases are kept until ``budget_seconds`` is reached; older
    ones are dropped, never cut.
    """
    source_rate, source = read_wav_mono16(reference_wav)
    parts = [resample_linear(source, source_rate, rate)]
    kept: list[tuple[str, bytes]] = []
    used = 0.0
    for text, pcm in reversed(previous):
        seconds = len(pcm) / 2 / rate
        if used + seconds > budget_seconds:
            break
        kept.append((text, pcm))
        used += seconds
    kept.reverse()
    gap = np.zeros(int(rate * gap_ms / 1000))
    for _, pcm in kept:
        parts.append(gap)
        parts.append(np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float64))
    audio = np.clip(np.round(np.concatenate(parts)), -32768, 32767).astype("<i2").tobytes()
    transcript = " ".join([reference_text.strip()] + [text.strip() for text, _ in kept])
    return wav_bytes(audio, rate), transcript, used


# --------------------------------------------------------------------- requests

def arm_texts(arm: str, reply: int) -> list[str]:
    """The request texts one arm sends for one reply."""
    split = ARMS[arm]["split"]
    if split == "short":
        return [SHORT_PHRASES[reply % len(SHORT_PHRASES)]]
    sentences = REPLIES[reply % len(REPLIES)]
    return [" ".join(sentences)] if split == "reply" else list(sentences)


def arm_reply_count(arm: str, replies: int) -> int:
    return len(SHORT_PHRASES) if ARMS[arm]["split"] == "short" else min(replies, len(REPLIES))


def build_request(arm: str, text: str, *, model: str, voice: str, seed: int, profile: str | None,
                  language: str | None, reply_id: str, reference: tuple[bytes, str] | None = None) -> dict[str, Any]:
    """The speech request body for one phrase of one arm."""
    spec = ARMS[arm]
    body: dict[str, Any] = {
        "model": model,
        "input": text,
        "response_format": "pcm",
        "stream": spec["delivery"] == "stream",
    }
    if language:
        body["language"] = language
    if profile:
        body["tuning"] = {"provider": PROVIDER, "scope": "realtime", "profile_id": profile, "overrides": {"seed": seed}}
    else:
        body["seed"] = seed
    if spec.get("continuation") == "emulated":
        if reference is None:
            raise ValueError("The emulated-continuation arm needs the voice's reference audio and transcript.")
        body["task_type"] = "Base"
        body["ref_audio"] = base64.b64encode(reference[0]).decode("ascii")
        body["ref_text"] = reference[1]
    else:
        body["voice"] = voice
    if spec.get("continuation") == "id":
        body["continuation_id"] = reply_id
    return body


class Server:
    """The few read-only calls and the speech call the benchmark makes."""

    def __init__(self, base_url: str, timeout: float = 240.0) -> None:
        import httpx  # imported here so the analysis helpers need only numpy

        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    def close(self) -> None:
        self.client.close()

    def get(self, path: str) -> Any:
        response = self.client.get(self.base + path)
        response.raise_for_status()
        return response.json()

    def status(self) -> dict[str, Any]:
        return self.get("/control/status")

    def voices(self) -> dict[str, Any]:
        return self.get("/v1/voices/profiles")

    def voice_profile(self, profile_id: str) -> dict[str, Any]:
        return self.get(f"/v1/voices/profiles/{profile_id}")

    def resolve(self, profile_id: str) -> dict[str, Any]:
        """Preview a tuning profile.  The route is a POST but stores nothing."""
        response = self.client.post(self.base + "/v1/tuning/resolve", json={
            "provider": PROVIDER, "scope": "realtime", "overrides": {}, "profile_id": profile_id})
        response.raise_for_status()
        return response.json()

    def outcome(self, request_id: str) -> dict[str, Any] | None:
        try:
            response = self.client.get(f"{self.base}/v1/audio/outcomes/{request_id}")
        except Exception:
            return None
        return response.json() if response.status_code == 200 else None

    def speak(self, body: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        first = None
        chunks: list[bytes] = []
        with self.client.stream("POST", self.base + "/v1/audio/speech", json=body) as response:
            headers = {key.lower(): value for key, value in response.headers.items()
                       if key.lower().startswith(("x-tts-", "x-audiocpp-", "content-type"))}
            if response.status_code != 200:
                detail = response.read().decode("utf-8", "replace")[:600]
                return {"status": response.status_code, "error": detail, "headers": headers,
                        "total_s": round(time.perf_counter() - started, 3), "pcm": b"", "rate": SAMPLE_RATE}
            for part in response.iter_bytes():
                if part and first is None:
                    first = time.perf_counter() - started
                chunks.append(part)
        pcm = b"".join(chunks)
        return {"status": 200, "error": None, "headers": headers, "pcm": pcm[: len(pcm) - len(pcm) % 2],
                "first_audio_s": None if first is None else round(first, 3),
                "total_s": round(time.perf_counter() - started, 3),
                "rate": int(headers.get("x-tts-sample-rate", SAMPLE_RATE))}


# -------------------------------------------------------------------------- run

def load_rows(out: Path) -> list[dict[str, Any]]:
    path = out / "results.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> int:
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        print(f"Unknown arm(s): {', '.join(unknown)}. Run the `arms` command to list them.", file=sys.stderr)
        return 2
    out = Path(args.out)
    server = Server(args.base_url)
    try:
        status = server.status()
        if status.get("state") != "loaded" or not status.get("activeModel"):
            print(f"No model is loaded (state: {status.get('state')}). Load one in Voice Studio first; "
                  "this benchmark never loads or unloads a model.", file=sys.stderr)
            return 3
        model = str(status["activeModel"])
        voice = args.voice
        if not voice:
            listed = server.voices()
            voice = listed.get("selectedVoice") or listed.get("defaultVoice")
        if not voice:
            print("The server has no clone voice. Create one in Voice Studio or pass --voice.", file=sys.stderr)
            return 3
        profile = None if args.profile == "none" else args.profile
        blocks = {"first_block_frames": None, "steady_block_frames": None}
        if profile:
            resolved = server.resolve(profile)
            values = {**(resolved.get("profile") or {}), **(resolved.get("temporaryOverrides") or {})}
            blocks = {key: values.get(key) for key in blocks}
        reference = None
        if any(ARMS[name].get("continuation") == "emulated" for name in arms):
            stored = server.voice_profile(voice.removeprefix("clone:"))
            reference = (base64.b64decode(stored["ref_audio"]), str(stored["ref_text"]))
        out.mkdir(parents=True, exist_ok=True)
        done = {(row["arm"], row["reply"], row["phrase"]) for row in load_rows(out) if row.get("status") == 200}
        run_id = time.strftime("%Y%m%d-%H%M%S")
        (out / "run.json").write_text(json.dumps({
            "run_id": run_id, "base_url": args.base_url, "model": model, "voice": voice, "profile": profile,
            "language": args.language, "engine_epoch": status.get("engineEpoch"), **blocks,
        }, indent=1), encoding="utf-8")
        with (out / "results.jsonl").open("a", encoding="utf-8") as results:
            for arm in arms:
                if not args.no_warmup:
                    warm = build_request(arm, WARMUP_TEXT, model=model, voice=voice, seed=1, profile=profile,
                                         language=args.language, reply_id=f"{run_id}-{arm}-warmup",
                                         reference=None if reference is None else
                                         (wav_bytes_from_reference(reference[0]), reference[1]))
                    server.speak(warm)
                for reply in range(arm_reply_count(arm, args.replies)):
                    seed = SEEDS[reply % len(SEEDS)]
                    previous: list[tuple[str, bytes]] = []
                    for phrase, text in enumerate(arm_texts(arm, reply)):
                        if (arm, reply, phrase) in done:
                            continue
                        extended = None
                        used = 0.0
                        if ARMS[arm].get("continuation") == "emulated" and reference is not None:
                            audio, transcript, used = continuation_reference(
                                reference[0], reference[1], previous, budget_seconds=args.continuation_seconds)
                            extended = (audio, transcript)
                        body = build_request(arm, text, model=model, voice=voice, seed=seed, profile=profile,
                                             language=args.language, reply_id=f"{run_id}-{arm}-{reply}",
                                             reference=extended)
                        result = server.speak(body)
                        pcm = result.pop("pcm")
                        row = {"arm": arm, "reply": reply, "phrase": phrase, "seed": seed, "text": text,
                               "chars": len(text), "delivery": ARMS[arm]["delivery"], "profile": profile,
                               "continuation_seconds_used": round(used, 3), **result}
                        if result["status"] == 200 and pcm:
                            relative = Path("audio") / arm / f"r{reply}_p{phrase}.wav"
                            (out / relative).parent.mkdir(parents=True, exist_ok=True)
                            (out / relative).write_bytes(wav_bytes(pcm, result["rate"]))
                            pitch, centroid = features(pcm, result["rate"])
                            row.update(wav=relative.as_posix(), audio_s=round(len(pcm) / 2 / result["rate"], 3),
                                       pitch_hz=rounded(pitch), centroid_hz=rounded(centroid))
                            previous.append((text, pcm))
                            request_id = result["headers"].get("x-tts-request-id")
                            if request_id:
                                row["outcome"] = server.outcome(request_id)
                        results.write(json.dumps(row) + "\n")
                        results.flush()
                        print(f"{arm} reply {reply} phrase {phrase}: status {row['status']}, "
                              f"first audio {row.get('first_audio_s')} s, audio {row.get('audio_s')} s")
    finally:
        server.close()
    return report(args)


def wav_bytes_from_reference(reference_wav: bytes) -> bytes:
    """The reference as the 24 kHz mono WAV the emulated arm sends."""
    return continuation_reference(reference_wav, "", [])[0]


# ----------------------------------------------------------------------- report

def read_pcm(out: Path, row: dict[str, Any]) -> tuple[int, bytes]:
    with wave.open(str(out / row["wav"]), "rb") as reader:
        return reader.getframerate(), reader.readframes(reader.getnframes())


def summarise_arm(out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [row for row in rows if row.get("status") == 200 and row.get("wav")]
    replies: dict[int, list[dict[str, Any]]] = {}
    for row in good:
        replies.setdefault(row["reply"], []).append(row)
    per_reply = []
    for number, items in sorted(replies.items()):
        items.sort(key=lambda row: row["phrase"])
        audio = b"".join(read_pcm(out, row)[1] for row in items)
        rate = int(items[0].get("rate") or SAMPLE_RATE)
        segments = segment_features(audio, rate)
        request_pitch = spread(row.get("pitch_hz") for row in items)
        request_centroid = spread(row.get("centroid_hz") for row in items)
        segment_pitch = spread(part["pitch_hz"] for part in segments)
        segment_centroid = spread(part["centroid_hz"] for part in segments)
        whole_pitch, _ = features(audio, rate)
        per_reply.append({
            "reply": number, "requests": len(items), "reply_pitch_hz": rounded(whole_pitch),
            "request_pitch_range": rounded(request_pitch[0]), "request_pitch_sd": rounded(request_pitch[1]),
            "request_centroid_range": rounded(request_centroid[0]), "request_centroid_sd": rounded(request_centroid[1]),
            "segment_pitch_range": rounded(segment_pitch[0]), "segment_pitch_sd": rounded(segment_pitch[1]),
            "segment_centroid_range": rounded(segment_centroid[0]), "segment_centroid_sd": rounded(segment_centroid[1]),
        })
    outcomes = [row["outcome"] for row in good if isinstance(row.get("outcome"), dict)]
    fit = linear_fit((item.get("generatedFrames") for item in outcomes), (item.get("talkerMs") for item in outcomes))
    counts: dict[str, dict[str, int]] = {"delivery_modes": {}, "decoder_modes": {}, "continuation": {}}
    for row in good:
        for key, header in (("delivery_modes", "x-tts-delivery-mode"), ("decoder_modes", "x-tts-decoder-mode"),
                            ("continuation", "x-tts-continuation")):
            value = row["headers"].get(header) or "(not reported)"
            counts[key][value] = counts[key].get(value, 0) + 1
    between = spread(item["reply_pitch_hz"] for item in per_reply)
    audio_seconds = sum(row.get("audio_s") or 0 for row in good)
    return {
        "replies": len(per_reply), "requests": len(rows), "failed": len(rows) - len(good),
        "failures": [{"reply": row["reply"], "phrase": row["phrase"], "status": row.get("status"),
                      "error": row.get("error")} for row in rows if row.get("status") != 200][:10],
        "first_audio_s_first_request": rounded(median(row.get("first_audio_s") for row in good if row["phrase"] == 0), 2),
        "first_audio_s_later_requests": rounded(median(row.get("first_audio_s") for row in good if row["phrase"] > 0), 2),
        "first_audio_s_p50": rounded(percentile((row.get("first_audio_s") for row in good), 0.5), 2),
        "first_audio_s_p95": rounded(percentile((row.get("first_audio_s") for row in good), 0.95), 2),
        "synthesis_s_per_audio_s": rounded(sum(row.get("total_s") or 0 for row in good) / audio_seconds, 2)
        if audio_seconds else None,
        "request_pitch_range": rounded(median(item["request_pitch_range"] for item in per_reply)),
        "request_pitch_sd": rounded(median(item["request_pitch_sd"] for item in per_reply)),
        "request_centroid_range": rounded(median(item["request_centroid_range"] for item in per_reply)),
        "segment_pitch_range": rounded(median(item["segment_pitch_range"] for item in per_reply)),
        "segment_pitch_sd": rounded(median(item["segment_pitch_sd"] for item in per_reply)),
        "segment_centroid_range": rounded(median(item["segment_centroid_range"] for item in per_reply)),
        "segment_centroid_sd": rounded(median(item["segment_centroid_sd"] for item in per_reply)),
        "between_reply_pitch_range": rounded(between[0]), "between_reply_pitch_sd": rounded(between[1]),
        "engine_ms": {key: rounded(median(item.get(key) for item in outcomes))
                      for key in ("promptMs", "prefillMs", "talkerMs", "decodeMs")},
        "talker_fixed_ms": None if fit is None else rounded(fit[0]),
        "talker_ms_per_frame": None if fit is None else rounded(fit[1], 2),
        **counts, "per_reply": per_reply,
    }


def summarise_pair(out: Path, first: list[dict[str, Any]], second: list[dict[str, Any]],
                   blocks: dict[str, Any]) -> dict[str, Any] | None:
    other = {(row["reply"], row["phrase"]): row for row in second if row.get("status") == 200 and row.get("wav")}
    found = []
    for row in first:
        match = other.get((row["reply"], row["phrase"]))
        if match is None or row.get("status") != 200 or not row.get("wav") or row["seed"] != match["seed"]:
            continue
        rate, left = read_pcm(out, row)
        _, right = read_pcm(out, match)
        boundaries = block_boundaries(int(blocks.get("first_block_frames") or 0),
                                      int(blocks.get("steady_block_frames") or 0), len(left) // 2)
        difference = pcm_difference(left, right, rate, boundaries)
        if difference is not None:
            found.append(difference)
    if not found:
        return None
    return {
        "compared": len(found),
        "identical": sum(1 for item in found if item["identical"]),
        "length_mismatches": sum(1 for item in found if item["length_delta_samples"]),
        "difference_db_below_signal": rounded(median(item["difference_db_below_signal"] for item in found)),
        "edge_ratio_median": rounded(median(item["edge_ratio"] for item in found), 2),
        "edge_ratio_max": rounded(max((item["edge_ratio"] for item in found if item["edge_ratio"] is not None),
                                      default=None), 2),
    }


def report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows = load_rows(out)
    if not rows:
        print(f"No results in {out}. Run the `run` command first.", file=sys.stderr)
        return 1
    run_file = out / "run.json"
    meta = json.loads(run_file.read_text(encoding="utf-8")) if run_file.is_file() else {}
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    summary: dict[str, Any] = {"run": meta, "arms": {name: summarise_arm(out, items) for name, items in by_arm.items()},
                               "pairs": {}}
    for first, second in PAIRS:
        if first in by_arm and second in by_arm:
            pair = summarise_pair(out, by_arm[first], by_arm[second], meta)
            if pair is not None:
                summary["pairs"][f"{first} vs {second}"] = pair
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    def show(value: Any) -> str:
        return "-" if value is None else str(value)

    print(f"\nmodel {meta.get('model')}  voice {meta.get('voice')}  profile {meta.get('profile')}")
    print("arm | requests (failed) | first audio s: first / later / p95 | "
          "pitch range Hz: per request / per segment | brightness range Hz per segment | talker fixed ms")
    for name, arm in summary["arms"].items():
        print(f"{name} | {arm['requests']} ({arm['failed']}) | {show(arm['first_audio_s_first_request'])} / "
              f"{show(arm['first_audio_s_later_requests'])} / {show(arm['first_audio_s_p95'])} | "
              f"{show(arm['request_pitch_range'])} / {show(arm['segment_pitch_range'])} | "
              f"{show(arm['segment_centroid_range'])} | {show(arm['talker_fixed_ms'])}")
    for name, pair in summary["pairs"].items():
        print(f"{name}: {pair['compared']} compared, {pair['identical']} identical, difference "
              f"{show(pair['difference_db_below_signal'])} dB below the signal, edge ratio median "
              f"{show(pair['edge_ratio_median'])} (max {show(pair['edge_ratio_max'])})")
    print(f"\nsummary written to {out / 'summary.json'}")
    return 1 if any(arm["failed"] for arm in summary["arms"].values()) else 0


def list_arms(_: argparse.Namespace) -> int:
    for name, spec in ARMS.items():
        default = " (default)" if name in DEFAULT_ARMS else ""
        print(f"{name}{default}: {spec['about']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("arms", help="List the request shapes the benchmark can run.").set_defaults(handler=list_arms)
    runner = commands.add_parser("run", help="Speak the replies and report.")
    runner.add_argument("--base-url", default=os.environ.get("AUDIO_CPP_BENCH_URL", DEFAULT_BASE_URL))
    runner.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    runner.add_argument("--replies", type=int, default=len(REPLIES))
    runner.add_argument("--voice", default="", help="clone:<id>; defaults to the server's selected voice.")
    runner.add_argument("--profile", default="balanced", help="Tuning profile id, or `none` for a bare request.")
    runner.add_argument("--language", default="English")
    runner.add_argument("--continuation-seconds", type=float, default=12.0)
    runner.add_argument("--no-warmup", action="store_true")
    runner.add_argument("--out", default=str(DEFAULT_OUT))
    runner.set_defaults(handler=run)
    reporter = commands.add_parser("report", help="Summarise results already on disk.")
    reporter.add_argument("--out", default=str(DEFAULT_OUT))
    reporter.set_defaults(handler=report)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
