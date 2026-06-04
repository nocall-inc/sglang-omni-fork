# SPDX-License-Identifier: Apache-2.0
"""Per-connection streaming TTS session.

One `StreamingTTSSession` instance owns the lifecycle of a single WebSocket
connection: receive control + text events, accumulate text until `stop`,
then run the synthesis pipeline and stream audio chunks back.

Uses the same internal pipeline (`build_speech_generate_request` +
`client.generate(stream=True)`) as the HTTP `/v1/audio/speech` endpoint with
`stream_format="audio"`. The only difference is the transport (WS msgpack
frames vs HTTP chunked PCM) and the protocol envelope (fish.audio-compatible
event-based messages).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import msgpack
from fastapi import WebSocket, WebSocketDisconnect

from sglang_omni.client import Client, ClientError
from sglang_omni.client.audio import encode_pcm

logger = logging.getLogger(__name__)

# Speculative prefill 設定: text delta が到着するたびに sgl-omni に "warm" リクエストを
# 投げて prefix cache (sglang radix cache) を温める。最終 stop 時の synthesis は
# prefill の大半がキャッシュ済みになり、TTFB が短縮される。
_WARM_DEBOUNCE_SEC = 0.25  # warm call 連打防止 (前回から 250ms 以上空ける)
_WARM_MIN_CHARS = 6        # 短すぎる text は warm 価値低い → skip


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
        self._text_parts: list[str] = []
        self._stopped = False
        self._closed = False
        # speculative prefill 状態
        self._last_warm_at: float = 0.0
        self._warm_task: asyncio.Task[None] | None = None

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
        # in-flight warm task を打ち切る
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()
            try:
                await self._warm_task
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
                await self._synthesize_and_stream()
                await self._send_event({"event": "finish"})
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
            self._text_parts.append(self._config.text_seed)
        logger.info(
            "session %s: start ref_id=%s latency=%s",
            self.session_id, self._config.reference_id, self._config.latency,
        )

    def _handle_text(self, payload: dict[str, Any]) -> None:
        text = payload.get("text")
        if isinstance(text, str) and text:
            self._text_parts.append(text)
            self._maybe_warm_prefix()

    def _maybe_warm_prefix(self) -> None:
        """Speculative prefill: fire-and-forget warm call to seed sglang's
        radix prefix cache so the final synth's prefill is mostly cached.

        Debounce 250ms。前 warm が in-flight でも cancel して新しいのを発火
        (常に最新 text で warm 状態を保つ)。レスポンスは捨てる。
        """
        if self._stopped or self._closed:
            return
        now = time.time()
        if now - self._last_warm_at < _WARM_DEBOUNCE_SEC:
            return
        full_text = "".join(self._text_parts).strip()
        if len(full_text) < _WARM_MIN_CHARS:
            return
        self._last_warm_at = now
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()
        self._warm_task = asyncio.create_task(self._warm_call(full_text))

    async def _warm_call(self, text: str) -> None:
        """sgl-omni に warm 用 generate を投げて radix cache を温める。

        max_new_tokens=1 で生成は最小、cancel されても問題ない。
        例外は debug log のみで握り潰す (warm 失敗で session を壊さない)。
        """
        try:
            from sglang_omni.serve.openai_api import build_speech_generate_request
            from sglang_omni.serve.protocol import CreateSpeechRequest, SpeechReference

            cfg = self._config or _SessionConfig()
            references_typed = None
            if cfg.references:
                references_typed = [SpeechReference(**ref) for ref in cfg.references]
            req_kwargs: dict[str, Any] = dict(
                model=self.model_name,
                input=text,
                response_format="pcm",
                speed=cfg.speed,
                stream=True,
                stream_format="audio",
                max_new_tokens=1,  # 最小生成、prefill のみが目的
            )
            if cfg.voice is not None:
                req_kwargs["voice"] = cfg.voice
            if cfg.reference_id is not None:
                req_kwargs["reference_id"] = cfg.reference_id
            if references_typed:
                req_kwargs["references"] = references_typed
            create_req = CreateSpeechRequest(**req_kwargs)
            gen_req = build_speech_generate_request(create_req, self.model_name)
            warm_id = f"{self.session_id}-warm-{uuid.uuid4().hex[:8]}"
            chunk_stream = self.client.generate(gen_req, request_id=warm_id)
            async for _chunk in chunk_stream:
                # 1 つ受信したら抜ける (prefill 完了の確証)
                return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("session %s: warm call failed: %s", self.session_id, e)

    async def _synthesize_and_stream(self) -> None:
        if self._config is None:
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
        """Run synthesis through the standard S2-Pro speech pipeline.

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
        chunk_stream = self.client.generate(gen_req, request_id=self.session_id)
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
