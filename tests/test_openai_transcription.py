from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from config import TranscriptionConfig
from exceptions import TranscriberError
from transcript_types import make_transcript_segment
import youtube_transcriber as yt


def test_transcribe_audio_openai_uses_configured_model_language_and_prompt(monkeypatch, tmp_path):
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"fake mp3 bytes")
    captured: dict[str, object] = {}

    class _FakeTranscriptions:
        def create(self, *, file, **request):
            captured["filename"] = file.name
            captured["request"] = request
            return "hello from openai"

    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=_FakeTranscriptions()))

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(yt, "_create_openai_client", lambda: fake_client)
    monkeypatch.setattr(yt, "_probe_duration_seconds", lambda file_path, ffmpeg_location: 12.5)

    config = TranscriptionConfig(
        openai_batch_model="gpt-4o-mini-transcribe",
        language="en",
        initial_prompt="Keep this faithful.",
        hotwords="Bryce, Codex",
    )

    transcript, segments = yt.transcribe_audio_openai(str(audio_path), config=config, cleanup_audio_file=False)

    assert transcript == "hello from openai"
    assert segments == [make_transcript_segment(start=0.0, end=12.5, text="hello from openai")]
    assert captured["filename"] == str(audio_path)
    assert captured["request"] == {
        "model": "gpt-4o-mini-transcribe",
        "response_format": "text",
        "language": "en",
        "prompt": "Keep this faithful.\nVocabulary and names that may appear: Bryce, Codex.",
    }
    assert audio_path.exists()


def test_openai_key_is_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(TranscriberError, match="OPENAI_API_KEY"):
        yt._require_openai_api_key()


def test_transcribe_audio_openai_preserves_caller_owned_file_by_default(monkeypatch, tmp_path):
    audio_path = tmp_path / "owned.mp3"
    audio_path.write_bytes(b"fake mp3 bytes")

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(yt, "_probe_duration_seconds", lambda file_path, ffmpeg_location: 1.0)
    monkeypatch.setattr(
        yt,
        "_transcribe_openai_chunks",
        lambda chunks, *, config, context: ("owned text", [make_transcript_segment(start=0.0, end=1.0, text="owned text")]),
    )

    transcript, segments = yt.transcribe_audio_openai(str(audio_path), config=TranscriptionConfig())

    assert transcript == "owned text"
    assert segments == [make_transcript_segment(start=0.0, end=1.0, text="owned text")]
    assert audio_path.exists()


def test_transcribe_audio_openai_removes_downloaded_file_when_requested(monkeypatch, tmp_path):
    audio_path = tmp_path / "downloaded.mp3"
    audio_path.write_bytes(b"fake mp3 bytes")

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(yt, "_probe_duration_seconds", lambda file_path, ffmpeg_location: 1.0)
    monkeypatch.setattr(
        yt,
        "_transcribe_openai_chunks",
        lambda chunks, *, config, context: ("downloaded text", [make_transcript_segment(start=0.0, end=1.0, text="downloaded text")]),
    )

    transcript, segments = yt.transcribe_audio_openai(str(audio_path), config=TranscriptionConfig(), cleanup_audio_file=True)

    assert transcript == "downloaded text"
    assert segments == [make_transcript_segment(start=0.0, end=1.0, text="downloaded text")]
    assert not audio_path.exists()


def test_prepare_openai_audio_chunks_reuses_small_supported_audio(monkeypatch, tmp_path):
    audio_path = tmp_path / "short.m4a"
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr(yt, "_probe_duration_seconds", lambda file_path, ffmpeg_location: 3.0)

    chunks = yt._prepare_openai_audio_chunks(
        str(audio_path),
        ffmpeg_location=None,
        temp_paths=[],
    )

    assert [(chunk.path, chunk.start, chunk.end) for chunk in chunks] == [(str(audio_path), 0.0, 3.0)]


def test_ffmpeg_executable_honors_custom_location_on_non_windows(monkeypatch, tmp_path):
    ffmpeg_dir = tmp_path / "ffmpeg-bin"
    ffmpeg_dir.mkdir()
    executable = ffmpeg_dir / "ffmpeg"
    executable.write_text("", encoding="utf-8")

    monkeypatch.setattr(yt.sys, "platform", "linux")

    assert yt._ffmpeg_executable("ffmpeg", str(ffmpeg_dir)) == str(executable)


def test_ffprobe_duration_timeout_returns_none_and_uses_media_timeout(monkeypatch, tmp_path):
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")
    captured: dict[str, object] = {}

    def _timeout_run(cmd, **kwargs):
        timeout = float(kwargs["timeout"])
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(yt.subprocess, "run", _timeout_run)

    assert yt._ffprobe_duration_seconds(str(audio_path), None) is None
    assert captured["timeout"] == yt.MEDIA_CMD_TIMEOUT
