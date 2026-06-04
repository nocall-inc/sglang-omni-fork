# SPDX-License-Identifier: Apache-2.0
"""Per-connection streaming TTS session.

One `StreamingTTSSession` instance owns the lifecycle of a single WebSocket
connection: receive control + text events, accumulate text until `stop`,
then run the synthesis pipeline and stream audio chunks back.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import msgpack
from fastapi import WebSocket, WebSocketDisconnect

from sglang_omni.client import Client, ClientError, GenerateRequest, Message, SamplingParams

logger = logging.getLogger(__name__)

# audio chunk granule for streaming responses. Matches the existing
# stream_format="audio" PCM chunked output.
_AUDIO_CHUNK_BYTES = 8192


@dataclass
class _SessionConfig:
    """Captured from the initial `start` event's `request` payload.

    Mirrors a subset of `CreateSpeechRequest` plus fish.audio-specific extras.
    Voice cloning + sampling params + response format are decided once per
    session (not per delta).
    """

    text_seed: str = ""  # any pre-supplied text in start.request.text
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
    chunk_length: int = 200  # fish.audio default; affects max prefill size
    latency: str = "balanced"  # low | balanced | normal


class StreamingTTSSession:
    """One WebSocket = one TTS session.

    Lifecycle:
      open()       — accept WS, init buffers
      run()        — recv loop: start → text* → stop, then synthesize + stream
      teardown()   — close WS, cleanup
    """

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
        self._text_parts: list[str] = []
        self._stopped = False
        self._closed = False

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
                await self._synthesize_and_stream()
                await self._send_event({"event": "finish"})
                return
            else:
                logger.warning("session %s: unknown event %r", self.session_id, event)

    def _extract_payload(self, msg: dict[str, Any]) -> Any:
        """Decode the inbound WS frame as msgpack dict."""
        if msg.get("type") == "websocket.receive":
            data = msg.get("bytes")
            if data is None:
                text = msg.get("text")
                if text is None:
                    return None
                # Optional JSON fallback for debugging
                import json
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return None
            try:
                return msgpack.unpackb(data, raw=False)
            except Exception:
                logger.warning("session %s: msgpack decode failed", self.session_id)
                return None
        return None

    def _handle_start(self, payload: dict[str, Any]) -> None:
        req = payload.get("request") or {}
        self._config = _SessionConfig(
            text_seed=req.get("text", "") or "",
            voice=req.get("voice"),
            reference_id=req.get("reference_id"),
            references=req.get("references") or [],
            response_format=req.get("response_format") or req.get("format") or "pcm",
            sample_rate=int(req.get("sample_rate") or 44100),
            speed=float(req.get("speed") or req.get("prosody", {}).get("speed") or 1.0),
            temperature=req.get("temperature"),
            top_p=req.get("top_p"),
            top_k=req.get("top_k"),
            repetition_penalty=req.get("repetition_penalty"),
            normalize=bool(req.get("normalize", True)),
            chunk_length=int(req.get("chunk_length") or 200),
            latency=req.get("latency") or "balanced",
        )
        if self._config.text_seed:
            self._text_parts.append(self._config.text_seed)
        logger.info(
            "session %s: start ref_id=%s latency=%s",
            self.session_id, self._config.reference_id, self._config.latency,
        )

    def _handle_text(self, payload: dict[str, Any]) -> None:
        text = payload.get("text")
        if isinstance(text, str) and text:
            self._text_parts.append(text)

    async def _synthesize_and_stream(self) -> None:
        if self._config is None:
            # client never sent start; default config
            self._config = _SessionConfig()
        full_text = "".join(self._text_parts).strip()
        if not full_text:
            await self._send_error("empty_text")
            return
        try:
            await self._stream_audio(full_text)
        except ClientError as exc:
            logger.exception("session %s: client error", self.session_id)
            await self._send_error(f"client_error:{exc}")
        except Exception as exc:
            logger.exception("session %s: synthesis error", self.session_id)
            await self._send_error(f"synth_error:{exc}")

    async def _stream_audio(self, text: str) -> None:
        """Run synthesis and stream audio chunks to the client.

        MVP: one call to client.generate() per session. Future enhancement:
        speculative prefill warming on each text_delta to reduce first-chunk
        latency.
        """
        cfg = self._config
        assert cfg is not None
        gen_req = self._build_generate_request(text)
        chunk_stream = self.client.generate(gen_req, request_id=self.session_id)
        async for chunk in chunk_stream:
            if chunk.audio_data is None:
                continue
            await self._send_audio_chunk(chunk, sample_rate=cfg.sample_rate)

    def _build_generate_request(self, text: str) -> GenerateRequest:
        """Translate session state → GenerateRequest for sgl-omni client."""
        cfg = self._config
        assert cfg is not None
        sampling = SamplingParams(
            temperature=cfg.temperature if cfg.temperature is not None else 1.0,
            top_p=cfg.top_p if cfg.top_p is not None else 1.0,
            top_k=cfg.top_k if cfg.top_k is not None else -1,
            repetition_penalty=cfg.repetition_penalty if cfg.repetition_penalty is not None else 1.0,
            max_tokens=2048,
        )
        # The actual GenerateRequest construction for S2-Pro TTS uses the
        # `speech` shorthand path. Reuse the existing client.speech() machinery
        # by routing through a synthetic chat-style request when needed.
        # NOTE: this is a placeholder; the actual implementation will use the
        # same internal pipeline that `_speech_audio_response` uses in
        # openai_api.py. For MVP we hand-build a GenerateRequest.
        msg = Message(role="user", content=text)
        return GenerateRequest(
            model=self.model_name,
            messages=[msg],
            sampling_params=sampling,
            modalities=["audio"],
            stream=True,
            extra={
                "voice": cfg.voice,
                "reference_id": cfg.reference_id,
                "references": cfg.references or None,
                "response_format": cfg.response_format,
                "sample_rate": cfg.sample_rate,
                "speed": cfg.speed,
            },
        )

    async def _send_audio_chunk(self, chunk: Any, *, sample_rate: int) -> None:
        """Wrap a GenerateChunk's audio into an `audio` event and send."""
        audio_bytes = chunk.audio_data
        if audio_bytes is None:
            return
        if not isinstance(audio_bytes, (bytes, bytearray)):
            # Some pipelines emit numpy; serialize as raw 16-bit PCM little-endian.
            try:
                import numpy as np
                arr = np.asarray(audio_bytes)
                if arr.dtype != np.int16:
                    arr = (arr * 32767.0).astype("<i2")
                audio_bytes = arr.tobytes()
            except Exception:
                logger.warning("session %s: failed to serialize audio chunk", self.session_id)
                return
        await self._send_event({
            "event": "audio",
            "audio": bytes(audio_bytes),
            "sample_rate": sample_rate,
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
