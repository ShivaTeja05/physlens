"""Speech to text with Meta's Muse Voice (muse-voice-transcribe-1.0), which is realtime-only.

Wire protocol (wss://api.meta.ai/v1/asr/realtime, as implemented by LiteLLM's Meta adapter):
  1. a JSON handshake with the key, audio encoding, model and mode (PUSH_TO_TALK: we say when the question ends);
  2. raw 16 kHz mono 16-bit PCM in 80 ms binary frames, at real-time pace;
  3. '{"type":"endStream"}' when the person lets go of the button.
It answers with "transcript" events (cumulative partials, then final=true).

`VoiceSession` relays a live microphone stream (the app's ask button); `transcribe_pcm` does the same for audio
already recorded.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Awaitable, Callable, Optional

URL = "wss://api.meta.ai/v1/asr/realtime"
RATE = 16_000
PACKET = RATE * 2 * 80 // 1000          # bytes in 80 ms
PRICE_PER_SECOND = 0.18 / 3600           # $0.18/hour as reported by third parties; not confirmed on Meta's page


def _handshake() -> str:
    return json.dumps({"authorization": {"accessToken": os.environ["META_API_KEY"]}, "audioEncoding": "PCM_16KHZ",
                       "model": os.environ.get("TRANSCRIBE_MODEL", "muse-voice-transcribe-1.0"), "mode": "PUSH_TO_TALK",
                       "partialMode": "CUMULATIVE", "emitAudioProgress": False, "languageBias": ["English"]})


class VoiceSession:
    """One push-to-talk question: feed PCM as it arrives, then finish() for the final text."""

    def __init__(self, on_partial: Callable[[str], Awaitable[None]]):
        self.on_partial = on_partial
        self.ws = None
        self.buf = b""
        self.sent = 0
        self.final: Optional[str] = None
        self.latest = ""
        self.reader: Optional[asyncio.Task] = None

    async def start(self) -> None:
        import websockets

        from . import budget
        budget.check(0.01)
        self.ws = await websockets.connect(URL, max_size=2 ** 22)
        await self.ws.send(_handshake())
        await asyncio.wait_for(self.ws.recv(), 10)          # {"sessionId": ...}
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for m in self.ws:
            if isinstance(m, bytes):
                continue
            e = json.loads(m)
            if e.get("type") in ("transcript", "speechComplete") and isinstance(e.get("transcript"), str):
                self.latest = e["transcript"]
                await self.on_partial(self.latest)
                if e.get("final") is True or e.get("type") == "speechComplete":
                    self.final = self.latest
            elif e.get("type") == "error":
                raise RuntimeError(f"Muse Voice error: {e}")

    async def feed(self, pcm: bytes) -> None:
        self.buf += pcm
        while len(self.buf) >= PACKET:
            await self.ws.send(self.buf[:PACKET])
            self.buf, self.sent = self.buf[PACKET:], self.sent + PACKET

    async def finish(self, timeout: float = 8.0) -> str:
        from . import budget
        if self.buf:
            await self.ws.send(self.buf)
            self.sent += len(self.buf)
            self.buf = b""
        await self.ws.send('{"type":"endStream"}')
        try:
            await asyncio.wait_for(self.reader, timeout)
        except (asyncio.TimeoutError, Exception):
            pass
        await self.ws.close()
        seconds = self.sent / (RATE * 2)
        budget.record("muse-voice-transcribe-1.0", seconds * PRICE_PER_SECOND, what="voice question", seconds=round(seconds, 2))
        return (self.final or self.latest).strip()


async def transcribe_pcm(pcm: bytes) -> str:
    """Recorded audio (16 kHz mono PCM16): streamed at real-time pace, as the service requires."""
    async def ignore(_: str) -> None:
        return None
    s = VoiceSession(ignore)
    await s.start()
    t0 = time.monotonic()
    for i in range(0, len(pcm), PACKET):
        await s.feed(pcm[i:i + PACKET])
        await asyncio.sleep(max(0.0, t0 + (i + PACKET) / (RATE * 2) - time.monotonic()))
    return await s.finish()
