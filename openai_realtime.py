"""OpenAI Realtime transcription WebSocket helpers."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any
from urllib.parse import quote

OPENAI_REALTIME_INPUT_SAMPLE_RATE = 24000
OPENAI_REALTIME_MIN_COMMIT_AUDIO_MS = 100
OPENAI_REALTIME_MIN_COMMIT_AUDIO_BYTES = int(OPENAI_REALTIME_INPUT_SAMPLE_RATE * 2 * OPENAI_REALTIME_MIN_COMMIT_AUDIO_MS / 1000)
DEFAULT_OPENAI_REALTIME_SESSION_MODEL = "gpt-realtime-1.5"
DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
_OPENAI_REALTIME_TRANSCRIPTION_MODEL_ALIASES = {
    "gpt-4o-transcribe-latest": DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL,
}
logger = logging.getLogger(__name__)


class OpenAIRealtimeConnectionClosed(RuntimeError):
    """Raised when the realtime WebSocket closes while the app is still streaming."""


def _exception_looks_like_closed_connection(exc: Exception) -> bool:
    class_name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "websocketconnectionclosed" in class_name
        or "connection to remote host was lost" in message
        or "connection is already closed" in message
        or "socket is already closed" in message
    )


def normalize_openai_realtime_transcription_model(
    value: object,
    default: str = DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL,
) -> str:
    """Normalize retired realtime transcription model aliases."""
    if not isinstance(value, str):
        return default

    model = value.strip()
    if not model:
        return default

    return _OPENAI_REALTIME_TRANSCRIPTION_MODEL_ALIASES.get(model, model)


def is_openai_api_configured() -> bool:
    """Return whether an OpenAI API key is available without exposing the key."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def require_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Rotate the exposed key and set a fresh key in the environment.")
    return api_key


class OpenAIRealtimeTranscriptionSession:
    """Small wrapper around the OpenAI Realtime transcription WebSocket."""

    def __init__(
        self,
        *,
        session_model: str,
        transcription_model: str,
        language: str | None = None,
        prompt: str | None = None,
        connection_timeout: float = 10.0,
        recv_timeout: float = 1.0,
    ) -> None:
        self.session_model = (
            session_model.strip()
            if isinstance(session_model, str) and session_model.strip()
            else DEFAULT_OPENAI_REALTIME_SESSION_MODEL
        )
        self.transcription_model = normalize_openai_realtime_transcription_model(transcription_model)
        self.language = language.strip() if language else None
        self.prompt = prompt.strip() if prompt else None
        self.connection_timeout = connection_timeout
        self.recv_timeout = recv_timeout
        self._ws: Any | None = None
        self._timeout_exception: type[BaseException] | None = None
        self.appended_pcm16_bytes = 0

    def describe_connection_state(self, exc: Exception | None = None) -> str:
        details: list[str] = []
        if exc is not None:
            details.append(f"exception={type(exc).__name__}: {exc}")
        if self._ws is None:
            details.append("ws=none")
            return "; ".join(details)

        for attr_name in ("connected", "status", "close_status", "close_reason"):
            value = getattr(self._ws, attr_name, None)
            if value is not None:
                details.append(f"ws.{attr_name}={value}")

        sock = getattr(self._ws, "sock", None)
        if sock is not None:
            for attr_name in ("connected", "status", "close_status", "close_reason"):
                value = getattr(sock, attr_name, None)
                if value is not None:
                    details.append(f"sock.{attr_name}={value}")

        return "; ".join(details) or "no websocket close details available"

    def connect(self) -> None:
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError("The websocket-client package is required for OpenAI realtime transcription.") from exc

        api_key = require_openai_api_key()
        url = f"wss://api.openai.com/v1/realtime?model={quote(self.session_model, safe='')}"
        self._timeout_exception = getattr(websocket, "WebSocketTimeoutException", TimeoutError)
        self._ws = websocket.create_connection(
            url,
            header=[f"Authorization: Bearer {api_key}"],
            timeout=self.connection_timeout,
        )
        self._ws.settimeout(self.recv_timeout)
        self._send_session_update()

    def _send_session_update(self) -> None:
        transcription: dict[str, Any] = {"model": self.transcription_model}
        if self.language:
            transcription["language"] = self.language
        if self.prompt:
            transcription["prompt"] = self.prompt

        self.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": OPENAI_REALTIME_INPUT_SAMPLE_RATE,
                            },
                            "noise_reduction": {"type": "near_field"},
                            "transcription": transcription,
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.6,
                                "prefix_padding_ms": 500,
                                "silence_duration_ms": 900,
                                "create_response": False,
                                "interrupt_response": False,
                            },
                        }
                    },
                },
            }
        )

    def send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime session is not connected.")
        try:
            self._ws.send(json.dumps(payload))
        except Exception as exc:
            if _exception_looks_like_closed_connection(exc):
                detail = self.describe_connection_state(exc)
                logger.warning("OpenAI realtime websocket closed while sending: %s", detail)
                raise OpenAIRealtimeConnectionClosed(detail) from exc
            raise

    def append_pcm16(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        self.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_bytes).decode("ascii"),
            }
        )
        self.appended_pcm16_bytes += len(pcm_bytes)

    def has_enough_audio_to_commit(self) -> bool:
        return self.appended_pcm16_bytes >= OPENAI_REALTIME_MIN_COMMIT_AUDIO_BYTES

    def commit(self) -> None:
        try:
            self.send_json({"type": "input_audio_buffer.commit"})
        except Exception:
            logger.exception("OpenAI realtime input_audio_buffer.commit failed in commit()")
            raise

    def recv_event(self) -> dict[str, Any] | None:
        if self._ws is None:
            return None
        try:
            raw = self._ws.recv()
        except Exception as exc:
            timeout_exception = self._timeout_exception
            if timeout_exception is not None and isinstance(exc, timeout_exception):
                return None
            if _exception_looks_like_closed_connection(exc):
                detail = self.describe_connection_state(exc)
                logger.warning("OpenAI realtime websocket closed while receiving: %s", detail)
                raise OpenAIRealtimeConnectionClosed(detail) from exc
            raise
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def close(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        finally:
            self._ws = None
