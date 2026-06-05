# SPDX-License-Identifier: Apache-2.0
"""Per-connection streaming TTS session.

One `StreamingTTSSession` instance owns the lifecycle of a single WebSocket
connection: receive control + text events, seal incoming text into segments at
sentence boundaries (or length fallback), and synthesize each sealed segment
in parallel with continued receive. Audio for each segment is emitted in
order via a single synth worker, so the client hears a contiguous stream.

This replaces the older "accumulate all text → synth once at stop" design.
TTFB ("last text byte sent" → "first audio byte received") shrinks because
synthesis of the first segment starts as soon as it seals — typically well
before the client's `stop`.

Reuses the standard internal pipeline (`build_speech_generate_request` +
`client.generate(stream=True)`) per segment so audio quality / feature parity
with the HTTP `/v1/audio/speech` path is preserved.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import msgpack
from fastapi import WebSocket, WebSocketDisconnect

from sglang_omni.client import Client, ClientError
from sglang_omni.client.audio import encode_pcm

logger = logging.getLogger(__name__)

# Parallel synth segmentation:
#   - 句読点で seal するのが基本
#   - 句読点なしで延々続く入力には _SEAL_LENGTH_FALLBACK で強制 seal
#   - 極短 segment は audio quality が落ちるので _MIN_SEAL_CHARS で merge
_SEAL_BOUNDARY_CHARS = "。！？!?.\n"
_SEAL_LENGTH_FALLBACK = 20
_MIN_SEAL_CHARS = 5


@dataclass
class _SessionConfig:
    """Captured from the initial `start` event's `request` payload.

    Mirrors a subset of `CreateSpeechRequest` plus fish.audio-specific extras.
    """

    text_seed: str = ""
    voice: str | None = None
    reference_id: str | None = None
    references: list[dict[str, Any]] = field(default_factory=list)
    response_format: str = "pcm"
    sample_rate: int = 44100
    speed: float = 1.0
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    normalize: bool = True
    chunk_length: int = 200
    latency: str = "balanced"


class StreamingTTSSession:
    """One WebSocket = one TTS session."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
    ) -> None:
        self.session_id = f"tts-stream-{uuid.uuid4()}"
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self._config: _SessionConfig | None = None
        self._pending: str = ""
        self._stopped = False
        self._closed = False
        # Parallel synth worker (起動は最初の segment seal 時 lazy)
        self._synth_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._synth_worker_task: asyncio.Task[None] | None = None
        self._worker_started = False
        self._worker_errored = False

    async def open(self) -> None:
        await self.websocket.accept()

    async def run(self) -> None:
        try:
            await self._recv_loop()
        except WebSocketDisconnect:
            logger.info("session %s: client disconnect", self.session_id)
        except Exception:
            logger.exception("session %s: unexpected error", self.session_id)
            await self._send_error("internal_error")
        finally:
            await self.teardown()

    async def teardown(self) -> None:
        if self._closed:
            return
        self._closed = True
        # in-flight synth worker を打ち切る (client 切断時など)
        if self._synth_worker_task is not None and not self._synth_worker_task.done():
            self._synth_worker_task.cancel()
            try:
                await self._synth_worker_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self.websocket.close()
        except Exception:
            pass

    async def _recv_loop(self) -> None:
        while not self._stopped:
            msg = await self.websocket.receive()
            payload = self._extract_payload(msg)
            if payload is None:
                continue
            event = payload.get("event") if isinstance(payload, dict) else None
            if event == "start":
                self._handle_start(payload)
            elif event == "text":
                self._handle_text(payload)
            elif event == "stop":
                self._stopped = True
                await self._on_stop()
                return
            else:
                logger.warning("session %s: unknown event %r", self.session_id, event)

    def _extract_payload(self, msg: dict[str, Any]) -> Any:
        """Decode the inbound WS frame as msgpack dict (binary) or JSON (text)."""
        if msg.get("type") != "websocket.receive":
            return None
        data = msg.get("bytes")
        if data is not None:
            try:
                return msgpack.unpackb(data, raw=False)
            except Exception:
                logger.warning("session %s: msgpack decode failed", self.session_id)
                return None
        text = msg.get("text")
        if text is not None:
            import json
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        return None

    def _handle_start(self, payload: dict[str, Any]) -> None:
        req = payload.get("request") or {}
        prosody = req.get("prosody") or {}
        self._config = _SessionConfig(
            text_seed=req.get("text", "") or "",
            voice=req.get("voice"),
            reference_id=req.get("reference_id"),
            references=req.get("references") or [],
            response_format=req.get("response_format") or req.get("format") or "pcm",
            sample_rate=int(req.get("sample_rate") or 44100),
            speed=float(req.get("speed") or prosody.get("speed") or 1.0),
            temperature=req.get("temperature"),
            top_p=req.get("top_p"),
            top_k=req.get("top_k"),
            repetition_penalty=req.get("repetition_penalty"),
            normalize=bool(req.get("normalize", True)),
            chunk_length=int(req.get("chunk_length") or 200),
            latency=req.get("latency") or "balanced",
        )
        if self._config.text_seed:
            self._pending += self._config.text_seed
            self._try_seal_segments()
        logger.info(
            "session %s: start ref_id=%s latency=%s",
            self.session_id, self._config.reference_id, self._config.latency,
        )

    def _handle_text(self, payload: dict[str, Any]) -> None:
        text = payload.get("text")
        if isinstance(text, str) and text:
            self._pending += text
            self._try_seal_segments()

    def _try_seal_segments(self) -> None:
        """Greedy seal at sentence boundaries (or length fallback).

        各 seal は worker queue に enqueue され、別 task で順次 synth される。
        """
        while True:
            seal_end = self._find_seal_end(self._pending)
            if seal_end < 0:
                return
            seg = self._pending[:seal_end]
            self._pending = self._pending[seal_end:]
            self._enqueue_segment(seg)

    @staticmethod
    def _find_seal_end(s: str) -> int:
        """Return end-index (exclusive) of the first valid seal point, or -1.

        Valid seal = sentence-end boundary char at position >= _MIN_SEAL_CHARS,
        or hard cutoff at _SEAL_LENGTH_FALLBACK chars.
        """
        for i, c in enumerate(s):
            if c in _SEAL_BOUNDARY_CHARS and (i + 1) >= _MIN_SEAL_CHARS:
                return i + 1
        if len(s) >= _SEAL_LENGTH_FALLBACK:
            return _SEAL_LENGTH_FALLBACK
        return -1

    def _enqueue_segment(self, seg: str) -> None:
        if not seg.strip():
            return
        self._synth_queue.put_nowait(seg)
        if not self._worker_started:
            self._worker_started = True
            self._synth_worker_task = asyncio.create_task(self._synth_worker())

    async def _on_stop(self) -> None:
        """Handle the `stop` event: flush pending → drain worker → finish."""
        # remaining pending text (たとえ < _MIN_SEAL_CHARS でも最終 segment として送る)
        if self._pending.strip():
            self._enqueue_segment(self._pending)
        self._pending = ""

        if not self._worker_started:
            # text が一度も来なかった
            await self._send_error("empty_text")
            return

        # sentinel + wait for worker drain
        await self._synth_queue.put(None)
        try:
            assert self._synth_worker_task is not None
            await self._synth_worker_task
        except (asyncio.CancelledError, Exception):
            logger.exception("session %s: worker await failed", self.session_id)

        if not self._worker_errored:
            await self._send_event({"event": "finish"})

    async def _synth_worker(self) -> None:
        """Pull sealed segments from queue, synth each, emit audio events in order.

        Single-worker = sequential synth = audio ordering preserved.
        Errors halt the worker (subsequent segments are dropped) and surface as
        an `error` event; `_on_stop` skips the `finish` emit in that case.
        """
        try:
            while True:
                seg = await self._synth_queue.get()
                if seg is None:
                    return
                try:
                    await self._stream_audio(seg)
                except ClientError as exc:
                    logger.exception("session %s: client error in segment", self.session_id)
                    await self._send_error(f"client_error:{exc}")
                    self._worker_errored = True
                    return
                except Exception as exc:
                    logger.exception("session %s: synth error in segment", self.session_id)
                    await self._send_error(f"synth_error:{exc}")
                    self._worker_errored = True
                    return
        except asyncio.CancelledError:
            raise

    async def _stream_audio(self, text: str) -> None:
        """Run synthesis on a single segment.

        Builds a CreateSpeechRequest with `stream=True, stream_format="audio"`,
        converts to a GenerateRequest, then iterates the client's async chunk
        stream emitting audio bytes back as msgpack `audio` events.

        Reuses `build_speech_generate_request` and the chunk handling from
        `_speech_audio_response` to ensure feature parity with the HTTP path.
        """
        cfg = self._config
        assert cfg is not None

        # Import locally to avoid circular imports at module-load time
        from sglang_omni.serve.openai_api import (
            _speech_pcm_chunk_bytes,
            build_speech_generate_request,
        )
        from sglang_omni.serve.protocol import CreateSpeechRequest, SpeechReference

        # Compose CreateSpeechRequest mirroring fish.audio's TTS knobs
        references_typed = None
        if cfg.references:
            references_typed = [SpeechReference(**ref) for ref in cfg.references]

        # CreateSpeechRequest はオプショナル None を許さない field があるので
        # set されたものだけ kwargs で渡す
        req_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            input=text,
            response_format="pcm",
            speed=cfg.speed,
            stream=True,
            stream_format="audio",
        )
        if cfg.voice is not None:
            req_kwargs["voice"] = cfg.voice
        if cfg.temperature is not None:
            req_kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            req_kwargs["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            req_kwargs["top_k"] = cfg.top_k
        if cfg.repetition_penalty is not None:
            req_kwargs["repetition_penalty"] = cfg.repetition_penalty
        if cfg.reference_id is not None:
            req_kwargs["reference_id"] = cfg.reference_id
        if references_typed:
            req_kwargs["references"] = references_typed
        create_req = CreateSpeechRequest(**req_kwargs)
        gen_req = build_speech_generate_request(create_req, self.model_name)

        # Stream chunks; same iteration pattern as _speech_audio_response
        emitted_samples = 0
        seg_id = f"{self.session_id}-seg-{uuid.uuid4().hex[:8]}"
        chunk_stream = self.client.generate(gen_req, request_id=seg_id)
        async for chunk in chunk_stream:
            if chunk.audio_data is None:
                continue
            audio_bytes, emitted_samples, sample_rate = _speech_pcm_chunk_bytes(
                chunk,
                emitted_samples=emitted_samples,
                speed=cfg.speed,
            )
            if audio_bytes is None:
                continue
            await self._send_event({
                "event": "audio",
                "audio": bytes(audio_bytes),
                "sample_rate": int(sample_rate),
            })

    async def _send_event(self, data: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            await self.websocket.send_bytes(msgpack.packb(data, use_bin_type=True))
        except Exception:
            logger.exception("session %s: send failed", self.session_id)

    async def _send_error(self, code: str) -> None:
        await self._send_event({"event": "error", "code": code})
