"""Tests for core transcription with mocked Whisper."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace


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

    def test_ensure_ffmpeg_on_path_adds_directory_once(self, monkeypatch):
        from youtube_transcriber import _ensure_ffmpeg_on_path

        monkeypatch.setenv("PATH", r"C:\Windows\System32")

        _ensure_ffmpeg_on_path(r"C:\ffmpeg\bin", context="test")
        _ensure_ffmpeg_on_path(r"C:\ffmpeg\bin", context="test")

        path_entries = os.environ["PATH"].split(os.pathsep)
        assert path_entries[0] == r"C:\ffmpeg\bin"
        assert path_entries.count(r"C:\ffmpeg\bin") == 1

    def test_load_whisper_pipeline_with_fallback_retries_on_cpu(self, monkeypatch):
        from youtube_transcriber import _load_whisper_pipeline_with_fallback

        calls: list[tuple[str, str]] = []

        def _fake_build(model_name: str, *, device: str, compute_type: str):
            calls.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("cublas missing")
            return "base-model", "pipeline"

        monkeypatch.setattr("youtube_transcriber._build_whisper_pipeline", _fake_build)

        base_model, pipeline, device, compute_type = _load_whisper_pipeline_with_fallback(
            "large-v3",
            device="cuda",
            compute_type="float16",
        )

        assert base_model == "base-model"
        assert pipeline == "pipeline"
        assert device == "cpu"
        assert compute_type == "int8"
        assert calls == [("cuda", "float16"), ("cpu", "int8")]

    def test_run_whisper_transcription_retries_on_cpu_and_collects_segments(self, monkeypatch):
        from youtube_transcriber import _WhisperExecutionState, _run_whisper_transcription
        from transcript_types import make_transcript_segment

        transcribe_calls: list[dict[str, object]] = []
        retry_markers: list[str] = []
        observed_text: list[str] = []

        class _FailingPipeline:
            def transcribe(self, input_source, **kwargs):
                transcribe_calls.append({"input_source": input_source, **kwargs})
                raise RuntimeError("cuda driver failure")

        successful_segments = [
            SimpleNamespace(start=0.0, end=1.5, text="hello world"),
        ]
        successful_info = SimpleNamespace(duration=90.0, language="en", language_probability=0.99)

        def _replacement_pipeline(model_name: str, *, device: str, compute_type: str):
            assert model_name == "large-v3"
            assert device == "cpu"
            assert compute_type == "int8"
            return (
                "cpu-base",
                SimpleNamespace(
                    transcribe=lambda input_source, **kwargs: (
                        transcribe_calls.append({"input_source": input_source, **kwargs}) or None,
                        (successful_segments, successful_info),
                    )[1]
                ),
            )

        monkeypatch.setattr("youtube_transcriber._build_whisper_pipeline", _replacement_pipeline)

        state = _WhisperExecutionState(
            model_name="large-v3",
            base_model="gpu-base",
            pipeline=_FailingPipeline(),
            device="cuda",
            compute_type="float16",
        )

        segments_data, info, updated_state = _run_whisper_transcription(
            state,
            "audio.wav",
            kwargs={"beam_size": 5},
            cpu_recovery_overrides={"beam_size": 1, "batch_size": 8},
            before_retry=lambda: retry_markers.append("retried"),
            segment_observer=lambda segment, info_obj: observed_text.append(
                f"{segment.text}:{info_obj.language}"
            ),
        )

        assert segments_data == [
            make_transcript_segment(start=0.0, end=1.5, text="hello world"),
        ]
        assert info is successful_info
        assert updated_state.device == "cpu"
        assert updated_state.compute_type == "int8"
        assert retry_markers == ["retried"]
        assert observed_text == ["hello world:en"]
        assert transcribe_calls == [
            {"input_source": "audio.wav", "beam_size": 5},
            {"input_source": "audio.wav", "beam_size": 1, "batch_size": 8},
        ]
