from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

import pytest

from openai_realtime import (
    OPENAI_REALTIME_MIN_COMMIT_AUDIO_BYTES,
    OpenAIRealtimeConnectionClosed,
    OpenAIRealtimeTranscriptionSession,
)


class _FakeTimeout(Exception):
    pass


class _FakeWebSocket:
    def __init__(self) -> None:
        self.recv_timeout: float | None = None
        self.sent: list[str] = []

    def settimeout(self, timeout: float) -> None:
        self.recv_timeout = timeout

    def send(self, payload: str) -> None:
        self.sent.append(payload)


def test_connect_uses_separate_connection_and_receive_timeouts(monkeypatch) -> None:
    fake_ws = _FakeWebSocket()
    captured: dict[str, object] = {}

    def _create_connection(url: str, *, header: list[str], timeout: float) -> _FakeWebSocket:
        captured["url"] = url
        captured["header"] = header
        captured["timeout"] = timeout
        return fake_ws

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "websocket",
        SimpleNamespace(
            WebSocketTimeoutException=_FakeTimeout,
            create_connection=_create_connection,
        ),
    )

    session = OpenAIRealtimeTranscriptionSession(
        session_model="gpt-realtime-test",
        transcription_model="gpt-4o-transcribe-test",
        connection_timeout=12.0,
        recv_timeout=2.5,
    )

    session.connect()

    assert captured["url"] == "wss://api.openai.com/v1/realtime?model=gpt-realtime-test"
    assert captured["timeout"] == 12.0
    assert fake_ws.recv_timeout == 2.5
    assert fake_ws.sent


def test_connect_normalizes_retired_transcription_model_alias(monkeypatch) -> None:
    fake_ws = _FakeWebSocket()
    captured: dict[str, object] = {}

    def _create_connection(url: str, *, header: list[str], timeout: float) -> _FakeWebSocket:
        captured["url"] = url
        return fake_ws

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "websocket",
        SimpleNamespace(
            WebSocketTimeoutException=_FakeTimeout,
            create_connection=_create_connection,
        ),
    )

    session = OpenAIRealtimeTranscriptionSession(
        session_model="",
        transcription_model="gpt-4o-transcribe-latest",
    )

    session.connect()

    sent_update = json.loads(fake_ws.sent[0])
    session_update = sent_update["session"]
    input_audio = session_update["audio"]["input"]
    transcription = input_audio["transcription"]
    assert captured["url"] == "wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5"
    assert sent_update["type"] == "session.update"
    assert session_update["type"] == "realtime"
    assert input_audio["format"] == {"type": "audio/pcm", "rate": 24000}
    assert transcription["model"] == "gpt-4o-transcribe"
    assert input_audio["turn_detection"]["create_response"] is False


def test_append_tracks_audio_available_for_commit() -> None:
    session = OpenAIRealtimeTranscriptionSession(
        session_model="gpt-realtime-test",
        transcription_model="gpt-4o-transcribe-test",
    )
    session._ws = _FakeWebSocket()

    assert session.has_enough_audio_to_commit() is False

    session.append_pcm16(b"0" * OPENAI_REALTIME_MIN_COMMIT_AUDIO_BYTES)

    assert session.has_enough_audio_to_commit() is True


def test_recv_event_treats_websocket_timeout_as_no_event() -> None:
    class _TimeoutWebSocket:
        def recv(self) -> str:
            raise _FakeTimeout("timed out")

    session = OpenAIRealtimeTranscriptionSession(
        session_model="gpt-realtime-test",
        transcription_model="gpt-4o-transcribe-test",
    )
    session._ws = _TimeoutWebSocket()
    session._timeout_exception = _FakeTimeout

    assert session.recv_event() is None


def test_recv_event_reports_connection_close_details() -> None:
    class _ClosedWebSocket:
        connected = False
        close_status = 1006
        close_reason = "abnormal closure"

        def recv(self) -> str:
            raise RuntimeError("Connection to remote host was lost.")

    session = OpenAIRealtimeTranscriptionSession(
        session_model="gpt-realtime-test",
        transcription_model="gpt-4o-transcribe-test",
    )
    session._ws = _ClosedWebSocket()

    with pytest.raises(OpenAIRealtimeConnectionClosed) as exc_info:
        session.recv_event()

    message = str(exc_info.value)
    assert "close_status=1006" in message
    assert "close_reason=abnormal closure" in message


def test_commit_logs_and_reraises_send_failures(caplog) -> None:
    class _FailingWebSocket:
        def send(self, payload: str) -> None:
            raise RuntimeError("network down")

    session = OpenAIRealtimeTranscriptionSession(
        session_model="gpt-realtime-test",
        transcription_model="gpt-4o-transcribe-test",
    )
    session._ws = _FailingWebSocket()

    with caplog.at_level(logging.ERROR, logger="openai_realtime"), pytest.raises(RuntimeError, match="network down"):
        session.commit()

    assert "input_audio_buffer.commit failed" in caplog.text
