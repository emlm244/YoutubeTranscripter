"""Tests for core transcription with mocked Whisper."""

from __future__ import annotations

import os
from pathlib import Path
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

    def test_build_runtime_transcribe_kwargs_clamps_cuda_batch_to_budget(self, monkeypatch):
        from config import TranscriptionConfig
        from youtube_transcriber import _build_runtime_transcribe_kwargs

        class _FakeProps:
            total_memory = 16 * 1024**3

        class _FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_properties(_index: int):
                return _FakeProps()

        class _FakeTorch:
            cuda = _FakeCuda()

        monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
        monkeypatch.setattr("youtube_transcriber._gpu_memory_fraction", lambda: 0.75)

        config = TranscriptionConfig(
            whisper_model="large-v3",
            beam_size=10,
            batch_size=32,
            patience=3.0,
            word_timestamps=True,
        )

        kwargs = _build_runtime_transcribe_kwargs(
            config,
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
            context="Unit test",
        )

        assert kwargs["beam_size"] == 10
        assert kwargs["batch_size"] == 12

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

    def test_runtime_config_logging_redacts_hotwords(self, caplog):
        from config import TranscriptionConfig
        from youtube_transcriber import _log_transcription_runtime_config

        with caplog.at_level("INFO", logger="youtube_transcriber"):
            _log_transcription_runtime_config(
                TranscriptionConfig(hotwords="Alpha, Beta"),
                context="Unit test",
            )

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "Alpha" not in messages
        assert "Beta" not in messages
        assert "2 hotword(s)" in messages

    def test_iter_segment_log_lines_expands_long_text(self):
        from youtube_transcriber import _iter_segment_log_lines

        lines = _iter_segment_log_lines(
            60.0,
            90.0,
            "One sentence. Two sentence? Three sentence! Four sentence continues with extra detail.",
        )

        assert lines[0].startswith("[001.0m -> 001.5m] One sentence.")
        assert any("Two sentence?" in line for line in lines[1:])
        assert any("Three sentence!" in line for line in lines[1:])
        assert any("Four sentence continues" in line for line in lines[1:])

    def test_collect_logged_segments_logs_continuation_lines(self, caplog):
        from youtube_transcriber import _collect_logged_segments

        segments = [
            SimpleNamespace(
                start=0.0,
                end=30.0,
                text="First sentence. Second sentence. Third sentence.",
            )
        ]
        info = SimpleNamespace(duration=30.0, language="en", language_probability=0.99)

        with caplog.at_level("INFO", logger="youtube_transcriber"):
            collected = _collect_logged_segments(segments, info=info)

        messages = [record.getMessage() for record in caplog.records]
        assert collected[0]["text"] == "First sentence. Second sentence. Third sentence."
        assert any(message.startswith("[000.0m -> 000.5m] First sentence.") for message in messages)
        assert any("Second sentence." in message for message in messages)
        assert any("Third sentence." in message for message in messages)

    def test_build_suspicion_retry_kwargs_disables_prompt_carryover(self):
        from youtube_transcriber import build_suspicion_retry_kwargs

        kwargs = {
            "initial_prompt": "Test prompt",
            "condition_on_previous_text": True,
            "no_speech_threshold": 0.3,
            "hallucination_silence_threshold": 0.5,
        }

        retry_kwargs = build_suspicion_retry_kwargs(kwargs)

        assert retry_kwargs["initial_prompt"] is None
        assert retry_kwargs["condition_on_previous_text"] is False
        assert retry_kwargs["no_speech_threshold"] == 0.5
        assert retry_kwargs["hallucination_silence_threshold"] == 1.0

    def test_transcription_result_looks_suspicious_for_prompt_leakage(self):
        from transcript_types import make_transcript_segment
        from youtube_transcriber import transcription_result_looks_suspicious

        segments = [
            make_transcript_segment(
                start=0.0,
                end=1.0,
                text="Do not paraphrase or rewrite for grammar. Use punctuation and capitalization only to make the spoken words readable.",
            )
        ]

        assert transcription_result_looks_suspicious(segments, input_duration=45.0) is True

    def test_transcription_result_looks_suspicious_for_single_hallucination(self):
        from transcript_types import make_transcript_segment
        from youtube_transcriber import transcription_result_looks_suspicious

        segments = [
            make_transcript_segment(
                start=0.0,
                end=1.0,
                text="Thank you for watching!",
            )
        ]

        assert transcription_result_looks_suspicious(segments, input_duration=45.0) is True

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

    def test_setup_device_respects_explicit_cpu_preference(self, monkeypatch):
        """An explicit CPU preference should bypass CUDA even when available."""
        from config import TranscriptionConfig
        from youtube_transcriber import _setup_device_and_compute_type

        monkeypatch.setattr("youtube_transcriber._ctranslate2_cuda_supported", lambda verbose=False: True)

        device, compute_type = _setup_device_and_compute_type(
            config=TranscriptionConfig(device_preference="cpu", compute_type="auto"),
            verbose=False,
        )

        assert device == "cpu"
        assert compute_type == "int8"

    def test_setup_device_coerces_float16_cpu_preference_to_int8(self, monkeypatch):
        from config import TranscriptionConfig
        from youtube_transcriber import _setup_device_and_compute_type

        monkeypatch.setattr("youtube_transcriber._ctranslate2_cuda_supported", lambda verbose=False: False)

        device, compute_type = _setup_device_and_compute_type(
            config=TranscriptionConfig(device_preference="cpu", compute_type="float16"),
            verbose=False,
        )

        assert device == "cpu"
        assert compute_type == "int8"

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

    def test_check_dependencies_can_skip_gpu_probe(self, monkeypatch):
        from youtube_transcriber import check_dependencies

        gpu_probe_calls = {"count": 0}

        def _get_whisper_cuda_status():
            gpu_probe_calls["count"] += 1
            return True, "GPU"

        monkeypatch.setattr("youtube_transcriber.find_ffmpeg", lambda: "C:\\ffmpeg")
        monkeypatch.setattr("youtube_transcriber.get_torch", lambda context: None)
        monkeypatch.setattr(
            "youtube_transcriber.get_whisper_cuda_status",
            _get_whisper_cuda_status,
        )

        class _DummyModule:
            WhisperModel = object

        monkeypatch.setitem(sys.modules, "yt_dlp", _DummyModule())
        monkeypatch.setitem(sys.modules, "faster_whisper", _DummyModule())
        monkeypatch.setitem(sys.modules, "sounddevice", object())

        all_ok, missing = check_dependencies(include_gpu_probe=False)

        assert all_ok is True
        assert missing == []
        assert gpu_probe_calls["count"] == 0

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

    def test_transcribe_local_file_uses_ranked_audio_stream_when_available(
        self, monkeypatch, tmp_path
    ):
        from config import TranscriptionConfig
        from transcript_types import make_transcript_segment
        from youtube_transcriber import (
            AudioStreamCandidate,
            AudioStreamInfo,
            _WhisperExecutionState,
            transcribe_local_file,
        )

        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"video")
        extracted_indices: list[int | None] = []
        rank_call_count = {"count": 0}

        execution_state = _WhisperExecutionState(
            model_name="large-v3",
            base_model="base",
            pipeline=object(),
            device="cpu",
            compute_type="int8",
        )
        segments = [make_transcript_segment(start=0.0, end=1.0, text="hello world")]

        monkeypatch.setattr("youtube_transcriber.get_torch", lambda context: None)
        monkeypatch.setattr("youtube_transcriber._log_transcription_runtime_config", lambda config, context: None)
        monkeypatch.setattr(
            "youtube_transcriber._initialize_whisper_execution",
            lambda *args, **kwargs: execution_state,
        )
        monkeypatch.setattr(
            "youtube_transcriber._build_runtime_transcribe_kwargs",
            lambda config, **kwargs: {"beam_size": 5, "batch_size": 8, "vad_filter": True},
        )
        monkeypatch.setattr("youtube_transcriber._probe_duration_seconds", lambda *args, **kwargs: 60.0)
        monkeypatch.setattr(
            "youtube_transcriber._extract_audio_to_wav",
            lambda _file_path, output_path, _ffmpeg, *, audio_index=None, **kwargs: (
                extracted_indices.append(audio_index),
                Path(output_path).write_bytes(b"wav"),
            ),
        )
        monkeypatch.setattr(
            "youtube_transcriber._maybe_preprocess_audio_path",
            lambda input_path, **kwargs: input_path,
        )
        monkeypatch.setattr(
            "youtube_transcriber._run_whisper_transcription",
            lambda state, input_source, **kwargs: (segments, SimpleNamespace(duration=60.0), state),
        )
        monkeypatch.setattr(
            "youtube_transcriber.transcription_result_looks_suspicious",
            lambda segment_data, input_duration=None: False,
        )
        monkeypatch.setattr(
            "youtube_transcriber._normalize_transcript_segments",
            lambda segment_data, **kwargs: ("hello world", list(segment_data), 0, 0),
        )

        def _fake_rank_audio_streams(*args, **kwargs):
            rank_call_count["count"] += 1
            return [
                AudioStreamCandidate(
                    info=AudioStreamInfo(
                        audio_index=1,
                        codec_name="aac",
                        channels=2,
                        sample_rate_hz=48000,
                        bit_rate_bps=192000,
                        language="en",
                        title="Speech",
                    ),
                    rms=0.4,
                    peak=0.9,
                    probes=((0.0, 0.4, 0.9),),
                )
            ]

        monkeypatch.setattr("youtube_transcriber._rank_audio_streams_for_transcription", _fake_rank_audio_streams)

        transcript, returned_segments = transcribe_local_file(
            str(video_file),
            ffmpeg_location="ffmpeg",
            config=TranscriptionConfig(),
        )

        assert transcript == "hello world"
        assert returned_segments == segments
        assert extracted_indices == [1]
        assert rank_call_count["count"] == 1

    def test_transcribe_local_file_tries_alternate_ranked_stream_after_initial_video_retries(
        self, monkeypatch, tmp_path
    ):
        from config import TranscriptionConfig
        from transcript_types import make_transcript_segment
        from youtube_transcriber import (
            AudioStreamCandidate,
            AudioStreamInfo,
            _WhisperExecutionState,
            transcribe_local_file,
        )

        video_file = tmp_path / "meeting.mp4"
        video_file.write_bytes(b"video")
        extracted_indices: list[int | None] = []
        transcribe_call_count = {"count": 0}
        rank_call_count = {"count": 0}

        execution_state = _WhisperExecutionState(
            model_name="large-v3",
            base_model="base",
            pipeline=object(),
            device="cpu",
            compute_type="int8",
        )
        recovered_segments = [
            make_transcript_segment(start=0.0, end=1.0, text="alternate track speech")
        ]

        monkeypatch.setattr("youtube_transcriber.get_torch", lambda context: None)
        monkeypatch.setattr("youtube_transcriber._log_transcription_runtime_config", lambda config, context: None)
        monkeypatch.setattr(
            "youtube_transcriber._initialize_whisper_execution",
            lambda *args, **kwargs: execution_state,
        )
        monkeypatch.setattr(
            "youtube_transcriber._build_runtime_transcribe_kwargs",
            lambda config, **kwargs: {"beam_size": 5, "batch_size": 8, "vad_filter": True},
        )
        monkeypatch.setattr("youtube_transcriber._probe_duration_seconds", lambda *args, **kwargs: 90.0)
        monkeypatch.setattr(
            "youtube_transcriber._extract_audio_to_wav",
            lambda _file_path, output_path, _ffmpeg, *, audio_index=None, **kwargs: (
                extracted_indices.append(audio_index),
                open(output_path, "wb").write(b"wav"),
            ),
        )
        monkeypatch.setattr(
            "youtube_transcriber._maybe_preprocess_audio_path",
            lambda input_path, **kwargs: input_path,
        )

        def _fake_run_whisper_transcription(state, input_source, **kwargs):
            transcribe_call_count["count"] += 1
            if transcribe_call_count["count"] <= 3:
                return [], SimpleNamespace(duration=90.0), state
            return recovered_segments, SimpleNamespace(duration=90.0), state

        monkeypatch.setattr("youtube_transcriber._run_whisper_transcription", _fake_run_whisper_transcription)
        monkeypatch.setattr(
            "youtube_transcriber.transcription_result_looks_suspicious",
            lambda segment_data, input_duration=None: False,
        )
        monkeypatch.setattr(
            "youtube_transcriber._normalize_transcript_segments",
            lambda segment_data, **kwargs: ("alternate track speech", list(segment_data), 0, 0),
        )

        def _fake_rank_audio_streams(*args, **kwargs):
            rank_call_count["count"] += 1
            return [
                AudioStreamCandidate(
                    info=AudioStreamInfo(
                        audio_index=0,
                        codec_name="aac",
                        channels=2,
                        sample_rate_hz=48000,
                        bit_rate_bps=192000,
                        language="en",
                        title="Muted",
                    ),
                    rms=0.2,
                    peak=0.4,
                    probes=((0.0, 0.2, 0.4),),
                ),
                AudioStreamCandidate(
                    info=AudioStreamInfo(
                        audio_index=1,
                        codec_name="aac",
                        channels=2,
                        sample_rate_hz=48000,
                        bit_rate_bps=128000,
                        language="en",
                        title="Speech",
                    ),
                    rms=0.1,
                    peak=0.3,
                    probes=((0.0, 0.1, 0.3),),
                ),
            ]

        monkeypatch.setattr("youtube_transcriber._rank_audio_streams_for_transcription", _fake_rank_audio_streams)

        transcript, returned_segments = transcribe_local_file(
            str(video_file),
            ffmpeg_location="ffmpeg",
            config=TranscriptionConfig(),
        )

        assert transcript == "alternate track speech"
        assert returned_segments == recovered_segments
        assert extracted_indices == [0, 1]
        assert rank_call_count["count"] == 1

    def test_transcribe_local_file_reuses_preloaded_execution_state(self, monkeypatch, tmp_path):
        from config import TranscriptionConfig
        from transcript_types import make_transcript_segment
        from youtube_transcriber import _WhisperExecutionState, transcribe_local_file

        audio_file = tmp_path / "clip.wav"
        audio_file.write_bytes(b"audio")

        reused_state = _WhisperExecutionState(
            model_name="large-v3",
            base_model=None,
            pipeline=object(),
            device="cpu",
            compute_type="int8",
        )
        observed_states: list[_WhisperExecutionState] = []
        segments = [make_transcript_segment(start=0.0, end=1.0, text="reused whisper transcript")]

        monkeypatch.setattr("youtube_transcriber.get_torch", lambda context: None)
        monkeypatch.setattr("youtube_transcriber._log_transcription_runtime_config", lambda config, context: None)
        monkeypatch.setattr(
            "youtube_transcriber._initialize_whisper_execution",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse the preloaded Whisper runtime")),
        )
        monkeypatch.setattr(
            "youtube_transcriber._build_runtime_transcribe_kwargs",
            lambda config, **kwargs: {"beam_size": 5, "batch_size": 8, "vad_filter": True},
        )
        monkeypatch.setattr("youtube_transcriber._probe_duration_seconds", lambda *args, **kwargs: 30.0)
        monkeypatch.setattr(
            "youtube_transcriber._maybe_preprocess_audio_path",
            lambda input_path, **kwargs: input_path,
        )
        monkeypatch.setattr(
            "youtube_transcriber._run_whisper_transcription",
            lambda state, input_source, **kwargs: (segments, SimpleNamespace(duration=30.0), state),
        )
        monkeypatch.setattr(
            "youtube_transcriber.transcription_result_looks_suspicious",
            lambda segment_data, input_duration=None: False,
        )
        monkeypatch.setattr(
            "youtube_transcriber._normalize_transcript_segments",
            lambda segment_data, **kwargs: ("reused whisper transcript", list(segment_data), 0, 0),
        )

        transcript, returned_segments = transcribe_local_file(
            str(audio_file),
            ffmpeg_location="ffmpeg",
            config=TranscriptionConfig(),
            execution_state=reused_state,
            execution_state_observer=observed_states.append,
        )

        assert transcript == "reused whisper transcript"
        assert returned_segments == segments
        assert observed_states == [reused_state]

    def test_transcribe_audio_rejects_retry_when_still_suspicious(self, monkeypatch, tmp_path):
        from config import TranscriptionConfig
        from transcript_types import make_transcript_segment
        from youtube_transcriber import _WhisperExecutionState, transcribe_audio

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"stub")

        suspicious_segments = [
            make_transcript_segment(
                start=0.0,
                end=1.0,
                text="Do not paraphrase or rewrite for grammar.",
            )
        ]
        execution_state = _WhisperExecutionState(
            model_name="large-v3",
            base_model="base",
            pipeline=object(),
            device="cpu",
            compute_type="int8",
        )
        run_calls = {"count": 0}
        normalized_inputs: list[list[dict[str, object]]] = []

        monkeypatch.setattr("youtube_transcriber.get_torch", lambda context: None)
        monkeypatch.setattr("youtube_transcriber._log_transcription_runtime_config", lambda config, context: None)
        monkeypatch.setattr(
            "youtube_transcriber._initialize_whisper_execution",
            lambda *args, **kwargs: execution_state,
        )
        monkeypatch.setattr(
            "youtube_transcriber._maybe_preprocess_audio_path",
            lambda audio_path, **kwargs: audio_path,
        )
        monkeypatch.setattr(
            "youtube_transcriber._build_runtime_transcribe_kwargs",
            lambda config, **kwargs: {"beam_size": 5, "batch_size": 8},
        )

        def _fake_run_whisper_transcription(state, input_source, **kwargs):
            run_calls["count"] += 1
            return suspicious_segments, SimpleNamespace(duration=45.0), state

        monkeypatch.setattr("youtube_transcriber._run_whisper_transcription", _fake_run_whisper_transcription)
        monkeypatch.setattr(
            "youtube_transcriber.transcription_result_looks_suspicious",
            lambda segments, input_duration=None: bool(segments),
        )

        def _fake_normalize_transcript_segments(segments_data, **kwargs):
            normalized_inputs.append(list(segments_data))
            return "", [], 0, 0

        monkeypatch.setattr(
            "youtube_transcriber._normalize_transcript_segments",
            _fake_normalize_transcript_segments,
        )

        transcript, segments = transcribe_audio(
            str(audio_file),
            config=TranscriptionConfig(),
            cleanup_audio_file=False,
        )

        assert run_calls["count"] == 2
        assert normalized_inputs == [[]]
        assert transcript == ""
        assert segments == []

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

        kwargs = {"beam_size": 5}
        segments_data, info, updated_state = _run_whisper_transcription(
            state,
            "audio.wav",
            kwargs=kwargs,
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
        assert kwargs == {"beam_size": 1, "batch_size": 8}
        assert transcribe_calls == [
            {"input_source": "audio.wav", "beam_size": 5},
            {"input_source": "audio.wav", "beam_size": 1, "batch_size": 8},
        ]

    def test_run_whisper_transcription_retries_on_cuda_with_smaller_batch(self):
        from youtube_transcriber import _WhisperExecutionState, _run_whisper_transcription
        from transcript_types import make_transcript_segment

        transcribe_calls: list[dict[str, object]] = []
        retry_markers: list[str] = []

        successful_segments = [
            SimpleNamespace(start=0.0, end=1.5, text="hello world"),
        ]
        successful_info = SimpleNamespace(duration=90.0, language="en", language_probability=0.99)

        class _FlakyPipeline:
            def __init__(self) -> None:
                self.calls = 0

            def transcribe(self, input_source, **kwargs):
                self.calls += 1
                transcribe_calls.append({"input_source": input_source, **kwargs})
                if self.calls == 1:
                    raise RuntimeError("CUDA failed with error out of memory")
                return successful_segments, successful_info

        state = _WhisperExecutionState(
            model_name="large-v3",
            base_model="gpu-base",
            pipeline=_FlakyPipeline(),
            device="cuda",
            compute_type="float16",
        )

        kwargs = {"beam_size": 10, "batch_size": 12}
        segments_data, info, updated_state = _run_whisper_transcription(
            state,
            "audio.wav",
            kwargs=kwargs,
            cpu_recovery_overrides={"beam_size": 1, "batch_size": 4},
            before_retry=lambda: retry_markers.append("retried"),
        )

        assert segments_data == [
            make_transcript_segment(start=0.0, end=1.5, text="hello world"),
        ]
        assert info is successful_info
        assert updated_state.device == "cuda"
        assert updated_state.compute_type == "float16"
        assert retry_markers == ["retried"]
        assert kwargs == {"beam_size": 10, "batch_size": 6}
        assert transcribe_calls == [
            {"input_source": "audio.wav", "beam_size": 10, "batch_size": 12},
            {"input_source": "audio.wav", "beam_size": 10, "batch_size": 6},
        ]

    def test_run_whisper_transcription_retries_when_generator_ooms_during_iteration(self, monkeypatch):
        from youtube_transcriber import _WhisperExecutionState, _run_whisper_transcription
        from transcript_types import make_transcript_segment

        transcribe_calls: list[dict[str, object]] = []
        retry_markers: list[str] = []
        observed_text: list[str] = []

        class _FailingGenerator:
            def __iter__(self):
                return self

            def __next__(self):
                raise RuntimeError("CUDA failed with error out of memory")

        class _GeneratorFailingPipeline:
            def transcribe(self, input_source, **kwargs):
                transcribe_calls.append({"input_source": input_source, **kwargs})
                return _FailingGenerator(), SimpleNamespace(duration=90.0, language="en", language_probability=0.99)

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
            pipeline=_GeneratorFailingPipeline(),
            device="cuda",
            compute_type="float16",
        )

        kwargs = {"beam_size": 5}
        segments_data, info, updated_state = _run_whisper_transcription(
            state,
            "audio.wav",
            kwargs=kwargs,
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
        assert kwargs == {"beam_size": 1, "batch_size": 8}
        assert transcribe_calls == [
            {"input_source": "audio.wav", "beam_size": 5},
            {"input_source": "audio.wav", "beam_size": 1, "batch_size": 8},
        ]
