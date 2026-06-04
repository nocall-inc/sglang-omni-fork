# SPDX-License-Identifier: Apache-2.0
"""Streaming TTS WebSocket endpoint compatible with fish.audio managed API.

Adds `/v1/tts/live` MessagePack-based WebSocket endpoint that mirrors
fish.audio's hosted streaming TTS protocol. Production livekit-agent that
currently points at `wss://api.fish.audio/v1/tts/live` can target a
self-hosted sgl-omni at `wss://<host>/v1/tts/live` without client changes.

Protocol (MessagePack):
  Client → Server:
    {"event": "start", "request": {"text": "", "reference_id": "...", ...}}
    {"event": "text", "text": "<incremental tokens>"}
    {"event": "stop"}                # flush, signals end of input
  Server → Client:
    {"event": "audio", "audio": <bytes>, "sample_rate": 44100}
    {"event": "finish"}              # synthesis done, server will close

MVP scope (current):
  - Buffer all text chunks until stop event
  - Submit full text to existing client.speech() pipeline
  - Stream audio bytes back as `audio` events
  - Single AR request per session

Future optimization (TODO):
  - Speculative prefill: warm sglang's prefix cache on each text_delta
  - True session continuation: maintain AR KV cache across text chunks
"""
