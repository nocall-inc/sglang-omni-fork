# SPDX-License-Identifier: Apache-2.0
"""Streaming TTS session manager — mirrors `realtime.manager.RealtimeSessionManager`.

Holds active sessions for the `/v1/tts/live` endpoint and provides open/close
hooks for the WebSocket handler in `openai_api.py`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket

from sglang_omni.client import Client
from sglang_omni.serve.streaming_tts.session import StreamingTTSSession

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StreamingTTSSessionManager:
    """Owns active streaming-TTS WebSocket sessions."""

    def __init__(self, *, client: Client, model_name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.sessions: dict[str, StreamingTTSSession] = {}

    def open(self, websocket: WebSocket) -> StreamingTTSSession:
        session = StreamingTTSSession(
            websocket,
            client=self.client,
            model_name=self.model_name,
        )
        self.sessions[session.session_id] = session
        logger.info("Streaming TTS session opened: %s", session.session_id)
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        await session.teardown()
        logger.info("Streaming TTS session closed: %s", session_id)

    def active_sessions(self) -> list[str]:
        return list(self.sessions.keys())
