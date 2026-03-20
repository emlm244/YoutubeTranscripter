"""Tests for core transcription with mocked Whisper."""

from __future__ import annotations

import sys


class TestTranscribeHelpers:
    """Test transcription helper functions."""

    def test_build_transcribe_kwargs(self):
        """Test kwargs builder produces all expected keys."""
        from youtube_transcriber import _build_transcribe_kwargs
        from config import TranscriptionConfig

        config = TranscriptionConfig(
            language="en",
            hotwords="Claude, RLHF",
            initial_prompt="Test prompt",
        )
        kwargs = _build_transcribe_kwargs(config)
        assert kwargs["language"] == "en"
        assert kwargs["hotwords"] == "Claude, RLHF"
        assert kwargs["initial_prompt"] == "Test prompt"
        assert kwargs["beam_size"] == 5
        assert kwargs["vad_filter"] is True
        assert "vad_parameters" in kwargs

    def test_build_vad_parameters(self):
        """Test VAD parameter builder."""
        from youtube_transcriber import _build_vad_parameters
        from config import TranscriptionConfig

        config = TranscriptionConfig(
            vad_threshold=0.30,
            min_speech_duration_ms=100,
            speech_pad_ms=500,
        )
        params = _build_vad_parameters(config)
        assert params["threshold"] == 0.30
        assert params["min_speech_duration_ms"] == 100
        assert params["speech_pad_ms"] == 500
        assert params["max_speech_duration_s"] == 30.0

    def test_setup_device_uses_ctranslate2_cuda_when_torch_cpu_only(self, monkeypatch):
        """Whisper should still use CUDA when CTranslate2 supports it."""
        from youtube_transcriber import _setup_device_and_compute_type

        monkeypatch.setattr("youtube_transcriber._ctranslate2_cuda_supported", lambda verbose=False: True)

        class _FakeCuda:
            @staticmethod
            def is_available():
                return False

        class _FakeTorch:
            cuda = _FakeCuda()

        monkeypatch.setattr("youtube_transcriber.get_torch", lambda context: _FakeTorch())

        device, compute_type = _setup_device_and_compute_type(verbose=False)
        assert device == "cuda"
        assert compute_type == "float16"

    def test_get_whisper_cuda_status_falls_back_to_ctranslate2_label(self, monkeypatch):
        """When torch is CPU-only, status should use CTranslate2 GPU metadata."""
        from youtube_transcriber import get_whisper_cuda_status

        monkeypatch.setattr("youtube_transcriber._ctranslate2_cuda_supported", lambda verbose=False: True)

        class _FakeCuda:
            @staticmethod
            def is_available():
                return False

        class _FakeTorch:
            cuda = _FakeCuda()

        class _FakeCTranslate2:
            @staticmethod
            def get_cuda_device_count():
                return 2

        monkeypatch.setattr("youtube_transcriber.get_torch", lambda context: _FakeTorch())
        monkeypatch.setitem(sys.modules, "ctranslate2", _FakeCTranslate2())

        available, name = get_whisper_cuda_status()
        assert available is True
        assert name == "CTranslate2 CUDA (2 devices)"

    def test_check_dependencies_treats_torch_as_optional(self, monkeypatch):
        from youtube_transcriber import check_dependencies

        monkeypatch.setattr("youtube_transcriber.find_ffmpeg", lambda: "C:\\ffmpeg")
        monkeypatch.setattr("youtube_transcriber.get_torch", lambda context: None)
        monkeypatch.setattr("youtube_transcriber.get_whisper_cuda_status", lambda: (False, ""))

        class _DummyModule:
            WhisperModel = object

        monkeypatch.setitem(sys.modules, "yt_dlp", _DummyModule())
        monkeypatch.setitem(sys.modules, "faster_whisper", _DummyModule())
        monkeypatch.setitem(sys.modules, "sounddevice", object())

        all_ok, missing = check_dependencies()

        assert all_ok is True
        assert missing == []

    def test_find_ffmpeg_uses_bundled_search_roots(self, monkeypatch, tmp_path):
        from youtube_transcriber import find_ffmpeg

        bundle_root = tmp_path / "bundle"
        bundle_root.mkdir()
        (bundle_root / "ffmpeg.exe").write_bytes(b"")
        (bundle_root / "ffprobe.exe").write_bytes(b"")

        monkeypatch.setattr("youtube_transcriber.get_ffmpeg_search_roots", lambda: [bundle_root])
        monkeypatch.setattr("youtube_transcriber._cached_ffmpeg_path", None)
        monkeypatch.setattr("youtube_transcriber._ffmpeg_cache_checked", False)

        ffmpeg_dir = find_ffmpeg()

        assert ffmpeg_dir == str(bundle_root)
