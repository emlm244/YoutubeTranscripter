"""Regression tests for the GUI logging bridge."""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from typing import Any, cast

import gui_transcriber as gt
import numpy as np

from config import GrammarConfig, TranscriptionConfig
from gui_transcriber import (
    QueueLogger,
    QueueHandler,
    TranscriberGUI,
    _apply_optional_grammar_corrections,
    _attach_worker_queue_logging,
    _build_transcription_complete_status,
    _detach_worker_queue_logging,
    _queue_transcript_snapshot,
    _worker_queue_bridge,
)
from transcript_types import make_transcript_segment


def _checkbox(value: bool):
    return SimpleNamespace(isChecked=lambda: value)


def _line_edit(value: str = ""):
    return SimpleNamespace(text=lambda: value)


def _combo(value: object):
    return SimpleNamespace(currentData=lambda: value)


def _spin(value: int | float):
    return SimpleNamespace(value=lambda: value)


def _build_fake_runtime_gui(config: TranscriptionConfig | None = None) -> Any:
    config = config or TranscriptionConfig()
    return cast(
        Any,
        SimpleNamespace(
            config=SimpleNamespace(
                transcription=config,
                grammar=GrammarConfig(),
                gpu_memory_fraction=0.9,
            ),
            preset_combo=_combo(gt.DEFAULT_PRESET),
            batch_backend_combo=_combo(config.batch_backend),
            openai_batch_model_combo=_combo(config.openai_batch_model),
            whisper_model_combo=_combo(config.whisper_model),
            language_input=_line_edit(config.language or ""),
            hotwords_input=_line_edit(config.hotwords or ""),
            device_preference_combo=_combo(config.device_preference),
            compute_type_combo=_combo(config.compute_type),
            batch_size_spin=_spin(config.batch_size),
            cpu_fallback_batch_spin=_spin(config.cpu_fallback_batch_size),
            beam_size_spin=_spin(config.beam_size),
            temperature_spin=_spin(config.temperature),
            vad_filter_checkbox=_checkbox(config.vad_filter),
            no_speech_threshold_spin=_spin(config.no_speech_threshold),
            word_timestamps_checkbox=_checkbox(config.word_timestamps),
            patience_spin=_spin(config.patience),
            length_penalty_spin=_spin(config.length_penalty),
            hallucination_silence_spin=_spin(config.hallucination_silence_threshold),
            vad_threshold_spin=_spin(config.vad_threshold),
            min_speech_duration_spin=_spin(config.min_speech_duration_ms),
            min_silence_duration_spin=_spin(config.min_silence_duration_ms),
            speech_pad_spin=_spin(config.speech_pad_ms),
            previous_text_checkbox=_checkbox(config.condition_on_previous_text),
            filler_cleanup_checkbox=_checkbox(config.clean_filler_words),
            filter_hallucinations_checkbox=_checkbox(config.filter_hallucinations),
            deduplicate_checkbox=_checkbox(config.deduplicate_repeated_segments),
            repetition_penalty_spin=_spin(config.repetition_penalty),
            no_repeat_ngram_spin=_spin(config.no_repeat_ngram_size),
            noise_reduction_checkbox=_checkbox(config.noise_reduction_enabled),
            normalize_audio_checkbox=_checkbox(config.normalize_audio),
            grammar_enhance_checkbox=_checkbox(False),
            grammar_backend_combo=_combo("auto"),
            grammar_language_input=_line_edit("en-US"),
            gector_batch_spin=_spin(8),
            gector_iterations_spin=_spin(5),
        ),
    )


def test_queue_logger_buffers_until_line_break() -> None:
    """Small stdout fragments should be coalesced before they hit the GUI queue."""
    target_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    logger = QueueLogger(target_queue)

    logger.write("Starting")
    logger.write(" transcription")
    assert target_queue.empty()

    logger.write("...\nNext")
    first = target_queue.get_nowait()
    assert first == ("progress", "Starting transcription...\n")
    assert target_queue.empty()

    logger.flush()
    second = target_queue.get_nowait()
    assert second == ("progress", "Next")
    assert target_queue.empty()


def test_queue_logger_normalizes_carriage_return_progress_updates() -> None:
    """tqdm-style carriage returns should not create blank progress messages."""
    target_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    logger = QueueLogger(target_queue)

    logger.write("\r 50%")
    assert target_queue.empty()

    logger.write("\r 60%")
    assert target_queue.get_nowait() == ("progress", " 50%\n")

    logger.flush()
    assert target_queue.get_nowait() == ("progress", " 60%")
    assert target_queue.empty()


def test_queue_logger_collapses_crlf_into_single_message() -> None:
    """Windows CRLF line endings should be forwarded as a single GUI update."""
    target_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    logger = QueueLogger(target_queue)

    logger.write("hello\r\nworld")
    assert target_queue.get_nowait() == ("progress", "hello\n")

    logger.flush()
    assert target_queue.get_nowait() == ("progress", "world")
    assert target_queue.empty()


def test_worker_queue_logging_temporarily_enables_backend_info_logs() -> None:
    """Worker loggers should emit INFO progress while a GUI job is active."""
    target_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    logger_names = ("youtube_transcriber", "grammar_postprocessor", "faster_whisper")
    original_levels = {name: logging.getLogger(name).level for name in logger_names}

    queue_handler, logger_states = _attach_worker_queue_logging(target_queue)
    try:
        for logger_name in logger_names:
            logger_obj = logging.getLogger(logger_name)
            assert logger_obj.level == logging.INFO
            assert queue_handler in logger_obj.handlers

        logging.getLogger("youtube_transcriber").info("Backend progress is visible")
        assert target_queue.get_nowait() == ("progress", "Backend progress is visible\n")
    finally:
        _detach_worker_queue_logging(queue_handler, logger_states)

    for logger_name in logger_names:
        logger_obj = logging.getLogger(logger_name)
        assert queue_handler not in logger_obj.handlers
        assert logger_obj.level == original_levels[logger_name]


def test_worker_queue_bridge_flushes_stdout_and_restores_logging(monkeypatch) -> None:
    """The shared worker bridge should always restore stdout and detach logging state."""
    target_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    detach_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target: ("handler", ["state"]))
    monkeypatch.setattr(
        gt,
        "_detach_worker_queue_logging",
        lambda handler, states: detach_calls.append((handler, states)),
    )

    original_stdout = gt.sys.stdout
    with _worker_queue_bridge(target_queue):
        print("bridged output", end="")

    assert gt.sys.stdout is original_stdout
    assert detach_calls == [("handler", ["state"])]
    assert target_queue.get_nowait() == ("progress", "bridged output")
    assert target_queue.empty()


def test_queue_handler_terminates_each_log_record_with_newline() -> None:
    """Logger records should remain visually separated in the GUI progress pane."""
    target_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    handler = QueueHandler(target_queue)

    logger = logging.getLogger("tmp-gui-queue-handler")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        logger.info("Line one")
        logger.info("Line two")
    finally:
        logger.removeHandler(handler)

    assert target_queue.get_nowait() == ("progress", "Line one\n")
    assert target_queue.get_nowait() == ("progress", "Line two\n")
    assert target_queue.empty()


def test_setup_backend_logging_for_gui_reuses_one_file_handler(monkeypatch, tmp_path) -> None:
    """Backend GUI logging should not create competing rotating handlers for one file."""
    logger_names = ("tmp-backend-a", "tmp-backend-b", "tmp-backend-c")
    logger_levels = {name: logging.INFO for name in logger_names}
    log_path = tmp_path / "youtube_transcriber.log"
    created_handlers: list[logging.Handler] = []

    class DummyFileHandler(logging.Handler):
        def __init__(self, base_filename: str) -> None:
            super().__init__()
            self.baseFilename = base_filename

        def emit(self, record: logging.LogRecord) -> None:
            return None

    original_logger_state = {
        name: (logging.getLogger(name).handlers[:], logging.getLogger(name).level)
        for name in logger_names
    }

    monkeypatch.setattr(gt, "BACKEND_LOGGER_LEVELS", logger_levels)
    monkeypatch.setattr(gt, "_backend_file_handler", None)
    monkeypatch.setattr(gt, "get_log_path", lambda _: log_path)

    def _build_handler(_: str) -> logging.Handler:
        handler = DummyFileHandler(str(log_path))
        created_handlers.append(handler)
        return handler

    monkeypatch.setattr(gt, "_build_rotating_file_handler", _build_handler)

    try:
        for logger_name in logger_names:
            logger_obj = logging.getLogger(logger_name)
            logger_obj.handlers.clear()
            logger_obj.setLevel(logging.NOTSET)

        gt.setup_backend_logging_for_gui()
        gt.setup_backend_logging_for_gui()

        assert len(created_handlers) == 1
        shared_handler = created_handlers[0]
        for logger_name in logger_names:
            logger_obj = logging.getLogger(logger_name)
            assert logger_obj.level == logging.INFO
            assert logger_obj.handlers.count(shared_handler) == 1
    finally:
        for logger_name in logger_names:
            logger_obj = logging.getLogger(logger_name)
            handlers, level = original_logger_state[logger_name]
            logger_obj.handlers[:] = handlers
            logger_obj.setLevel(level)


def test_queue_transcript_snapshot_formats_timestamped_output() -> None:
    """Shared snapshot helper should keep segment data and rendered text aligned."""
    target_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    segments = [make_transcript_segment(start=0.0, end=1.25, text="hello world")]

    _queue_transcript_snapshot(
        target_queue,
        "hello world",
        segments,
        output_format="timestamped",
    )

    assert target_queue.get_nowait() == ("segments", segments)
    assert target_queue.get_nowait() == (
        "transcript",
        gt.format_transcript_with_timestamps(segments),
    )
    assert target_queue.empty()


def test_apply_optional_grammar_corrections_reports_success(monkeypatch) -> None:
    """Grammar helper should centralize queue messages and return the processed snapshot."""
    target_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
    raw_segments = [make_transcript_segment(start=0.0, end=1.0, text="raw text")]
    processed_segments = [make_transcript_segment(start=0.0, end=1.0, text="fixed text")]

    monkeypatch.setattr(
        gt,
        "post_process_grammar",
        lambda **kwargs: ("fixed text", processed_segments, True),
    )

    result = _apply_optional_grammar_corrections(
        target_queue,
        "raw text",
        raw_segments,
        GrammarConfig(enabled=True),
        warning_color="#f90",
        start_status="Applying grammar corrections...",
    )

    assert result.completed is True
    assert result.grammar_enhanced is True
    assert result.transcript == "fixed text"
    assert result.segments_data == processed_segments
    assert _drain_queue(target_queue) == [
        ("status", "Applying grammar corrections...", "#f90"),
        ("progress", "Fixing grammar...\n"),
        ("progress", "Grammar correction applied.\n"),
    ]


def test_build_transcription_complete_status_variants() -> None:
    """Completion status helper should preserve the repo's existing wording."""
    assert _build_transcription_complete_status(grammar_enhanced=False) == "Transcription complete!"
    assert (
        _build_transcription_complete_status(grammar_enhanced=True, segment_count=3)
        == "Transcription complete! (3 segments) (grammar corrected)"
    )
    assert (
        _build_transcription_complete_status(grammar_enhanced=True, word_count=42)
        == "Transcription complete (42 words) - grammar corrected"
    )


def test_update_grammar_status_uses_current_checkbox_config(monkeypatch) -> None:
    text_updates: list[str] = []
    style_updates: list[str] = []

    fake_gui = cast(
        Any,
        SimpleNamespace(
            grammar_enhance_checkbox=_checkbox(True),
            grammar_status_label=SimpleNamespace(
                setText=lambda text: text_updates.append(text),
                setStyleSheet=lambda style: style_updates.append(style),
            ),
            theme=SimpleNamespace(colors=SimpleNamespace(text_secondary="#999", success="#0f0", info="#09f", warning="#ff0")),
            _build_grammar_config=lambda: GrammarConfig(enabled=True, backend="gector"),
        ),
    )
    monkeypatch.setattr(gt, "check_grammar_status", lambda **kwargs: (True, "GECToR (ready on demand)"))

    TranscriberGUI._update_grammar_status(fake_gui)

    assert text_updates[-1] == "(GECToR (ready on demand))"
    assert style_updates


def test_build_transcription_config_keeps_audio_cleanup_toggles_independent(monkeypatch) -> None:
    """Noise reduction and normalization should remain independently configurable."""
    fake_gui = _build_fake_runtime_gui(
        TranscriptionConfig(
            clean_filler_words=True,
            noise_reduction_enabled=False,
            normalize_audio=True,
        )
    )
    monkeypatch.setattr(gt, "apply_preset", lambda config, preset_name: None)

    config = TranscriberGUI._build_transcription_config(fake_gui)

    assert config.noise_reduction_enabled is False
    assert config.normalize_audio is True


def test_build_transcription_config_syncs_live_gpu_memory_fraction(monkeypatch) -> None:
    """The GPU memory spinbox should update the shared runtime config before Whisper loads."""
    fake_gui = _build_fake_runtime_gui()
    fake_gui.gpu_memory_spin = _spin(0.75)
    monkeypatch.setattr(gt, "apply_preset", lambda config, preset_name: None)

    TranscriberGUI._build_transcription_config(fake_gui)

    assert fake_gui.config.gpu_memory_fraction == 0.75


def test_build_transcription_config_clears_hotwords_when_input_blank(monkeypatch) -> None:
    """Clearing the hotwords field should not keep stale hints from saved config."""
    fake_gui = _build_fake_runtime_gui(TranscriptionConfig(hotwords="Old Hint"))
    fake_gui.hotwords_input = _line_edit("")
    monkeypatch.setattr(gt, "apply_preset", lambda config, preset_name: None)

    config = TranscriberGUI._build_transcription_config(fake_gui)

    assert config.hotwords is None


def test_save_settings_persists_runtime_and_ui_config(monkeypatch) -> None:
    """Closing the app should persist the current UI selections and runtime knobs."""
    saved_values: dict[str, object] = {}
    saved_config_calls: list[bool] = []
    transcription_config = TranscriptionConfig(whisper_model="large-v3", batch_size=16)
    grammar_config = GrammarConfig(enabled=True, backend="languagetool")

    class _FakeQSettings:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def setValue(self, key: str, value: object) -> None:
            saved_values[key] = value

    fake_gui = cast(
        Any,
        SimpleNamespace(
            content_splitter=SimpleNamespace(saveState=lambda: b"splitter-state"),
            url_input=_line_edit("https://youtube.com/watch?v=dQw4w9WgXcQ"),
            format_combo=_combo("timestamped"),
            mic_combo=SimpleNamespace(currentText=lambda: "Studio Mic"),
            preset_combo=_combo("balanced"),
            _build_transcription_config=lambda: transcription_config,
            _build_grammar_config=lambda: grammar_config,
            gpu_memory_spin=_spin(0.75),
            config=SimpleNamespace(
                ui=SimpleNamespace(last_youtube_url="", output_format="plain", transcription_preset="max_accuracy"),
                recording=SimpleNamespace(default_microphone=""),
                transcription=TranscriptionConfig(),
                grammar=GrammarConfig(),
                gpu_memory_fraction=0.90,
            ),
        ),
    )

    monkeypatch.setattr(gt.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(gt, "save_config", lambda: saved_config_calls.append(True))

    TranscriberGUI._save_settings(fake_gui)

    assert saved_values["splitter/state"] == b"splitter-state"
    assert fake_gui.config.ui.last_youtube_url.endswith("dQw4w9WgXcQ")
    assert fake_gui.config.ui.output_format == "timestamped"
    assert fake_gui.config.ui.transcription_preset == "balanced"
    assert fake_gui.config.recording.default_microphone == "Studio Mic"
    assert fake_gui.config.transcription == transcription_config
    assert fake_gui.config.grammar == grammar_config
    assert fake_gui.config.gpu_memory_fraction == 0.75
    assert saved_config_calls == [True]


def test_transcription_setting_help_texts_cover_visible_titles() -> None:
    """Every visible settings title should have help copy for the ? badges."""
    expected_keys = {
        "Preset:",
        "Backend:",
        "OpenAI Model:",
        "Model:",
        "Language:",
        "Hotwords:",
        "Grammar Fix",
        "Clean Fillers",
        "Noise Reduction",
        "Normalize Audio",
        "Filter Hallucinations",
        "Deduplicate Loops",
        "Word Timestamps",
        "Context Carryover",
        "Use VAD",
        "Device:",
        "Compute:",
        "Batch:",
        "CPU Fallback Batch:",
        "Beam:",
        "Temperature:",
        "GPU Memory %:",
        "No-Speech Threshold:",
        "VAD Threshold:",
        "Min Speech (ms):",
        "Min Silence (ms):",
        "Speech Pad (ms):",
        "Repetition Penalty:",
        "No-Repeat N-gram:",
        "Patience:",
        "Length Penalty:",
        "Hallucination Silence:",
        "Grammar Backend:",
        "Grammar Language:",
        "GECToR Batch:",
        "GECToR Iterations:",
    }

    assert expected_keys.issubset(gt.TRANSCRIPTION_SETTING_HELP_TEXTS)
    assert all(gt.TRANSCRIPTION_SETTING_HELP_TEXTS[key].strip() for key in expected_keys)


def test_start_recording_local_backend_does_not_require_openai_key(monkeypatch) -> None:
    """Local Whisper microphone recording should start without an OpenAI key."""
    progress_messages: list[str] = []
    status_calls: list[tuple[str, object]] = []

    fake_gui = cast(
        Any,
        SimpleNamespace(
            is_recording=False,
            transcribe_thread=None,
            append_progress=lambda text: progress_messages.append(text),
            update_status=lambda message, color=None: status_calls.append((message, color)),
            theme=SimpleNamespace(colors=SimpleNamespace(error="#f00", warning="#f90")),
            _build_transcription_config=lambda: TranscriptionConfig(batch_backend="local_whisper"),
            _start_recording_now=lambda: status_calls.append(("started", None)),
        ),
    )
    monkeypatch.setattr(gt, "is_openai_api_configured", lambda: False)

    TranscriberGUI.start_recording(fake_gui)

    assert progress_messages == []
    assert status_calls == [("started", None)]


def test_start_recording_blocks_when_openai_key_missing(monkeypatch) -> None:
    progress_messages: list[str] = []
    status_calls: list[tuple[str, object]] = []

    fake_gui = cast(
        Any,
        SimpleNamespace(
            is_recording=False,
            transcribe_thread=None,
            append_progress=lambda text: progress_messages.append(text),
            update_status=lambda message, color=None: status_calls.append((message, color)),
            theme=SimpleNamespace(colors=SimpleNamespace(error="#f00", warning="#f90")),
            _build_transcription_config=lambda: TranscriptionConfig(batch_backend="openai"),
            _start_recording_now=lambda: status_calls.append(("started", None)),
        ),
    )
    monkeypatch.setattr(gt, "is_openai_api_configured", lambda: False)

    TranscriberGUI.start_recording(fake_gui)

    assert "OPENAI_API_KEY is not set" in progress_messages[0]
    assert status_calls == [("Missing OPENAI_API_KEY", "#f00")]


def test_start_recording_waits_for_existing_transcription() -> None:
    """Recording should not start while another transcription worker is still active."""
    progress_messages: list[str] = []
    status_calls: list[tuple[str, object]] = []

    fake_gui = cast(
        Any,
        SimpleNamespace(
            is_recording=False,
            transcribe_thread=SimpleNamespace(is_alive=lambda: True),
            append_progress=lambda text: progress_messages.append(text),
            update_status=lambda message, color=None: status_calls.append((message, color)),
            theme=SimpleNamespace(colors=SimpleNamespace(warning="#f90")),
        ),
    )

    TranscriberGUI.start_recording(fake_gui)

    assert progress_messages == [
        "Wait for the current transcription to finish before starting a new recording.\n"
    ]
    assert status_calls == [("Transcription already in progress", "#f90")]


def test_process_queue_recording_captured_starts_batch_transcription() -> None:
    """A completed capture should reset capture UI and hand audio to the batch worker."""
    started_jobs: list[tuple[list[float], int, int]] = []
    reset_calls: list[str] = []
    status_calls: list[tuple[str, object]] = []

    fake_gui = cast(
        Any,
        SimpleNamespace(
            output_queue=queue.Queue(),
            config=SimpleNamespace(recording=SimpleNamespace(sample_rate=16000)),
            theme=SimpleNamespace(colors=SimpleNamespace(warning="#f90")),
            _reset_recording_ui_state=lambda: reset_calls.append("reset"),
            update_status=lambda message, color=None: status_calls.append((message, color)),
            _start_recorded_audio_transcription=lambda *, audio_buffer, duration_seconds, sample_rate: started_jobs.append(
                (audio_buffer, duration_seconds, sample_rate)
            ),
        ),
    )
    fake_gui.output_queue.put(("recording_captured", [0.1, -0.2], 3, 48000))

    TranscriberGUI.process_queue(fake_gui)

    assert reset_calls == ["reset"]
    assert status_calls == [("Transcribing microphone recording...", "#f90")]
    assert started_jobs == [([0.1, -0.2], 3, 48000)]


def test_transcribe_recorded_audio_thread_uses_shared_batch_and_cleans_temp(tmp_path: Path) -> None:
    """Recorded microphone audio should be persisted temporarily, batch-transcribed, then removed."""
    temp_audio = tmp_path / "recording.wav"
    temp_audio.write_bytes(b"wav")
    batch_calls: list[str] = []

    fake_gui = cast(
        Any,
        SimpleNamespace(
            output_queue=queue.Queue(),
            theme=SimpleNamespace(colors=SimpleNamespace(warning="#f90", success_light="#0f0")),
            _write_recorded_audio_to_temp_wav=lambda audio_buffer, sample_rate: str(temp_audio),
            _transcribe_batch_media_file=lambda **kwargs: batch_calls.append(kwargs["file_path"]) or True,
        ),
    )

    TranscriberGUI.transcribe_recorded_audio_thread(
        fake_gui,
        [0.1, -0.1],
        2,
        16000,
        TranscriptionConfig(batch_backend="local_whisper"),
        GrammarConfig(enabled=False),
    )

    assert batch_calls == [str(temp_audio)]
    assert not temp_audio.exists()
    assert ("microphone_done", True) in _drain_queue(fake_gui.output_queue)


def test_microphone_compare_mode_replaces_transcript_and_clears_segments(monkeypatch) -> None:
    """Compare output should use local-file backend semantics for microphone recordings."""
    queued_snapshots: list[tuple[str, object]] = []
    monkeypatch.setattr(gt, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(gt, "is_openai_api_configured", lambda: True)
    monkeypatch.setattr(
        gt,
        "transcribe_local_file_openai",
        lambda **kwargs: ("openai text", [make_transcript_segment(start=0.0, end=1.0, text="openai text")]),
    )
    monkeypatch.setattr(
        gt,
        "transcribe_local_file",
        lambda **kwargs: ("local text", [make_transcript_segment(start=0.0, end=1.0, text="local text")]),
    )
    monkeypatch.setattr(
        gt,
        "_queue_transcript_snapshot",
        lambda target_queue, transcript, segments_data, **kwargs: queued_snapshots.append((transcript, segments_data)),
    )
    monkeypatch.setattr(gt, "unload_gector", lambda: None)
    monkeypatch.setattr(gt, "get_torch", lambda context: None)

    fake_gui = cast(
        Any,
        SimpleNamespace(
            output_queue=queue.Queue(),
            theme=SimpleNamespace(colors=SimpleNamespace(warning="#f90", success_light="#0f0")),
            _build_reusable_whisper_execution_state=lambda transcription_config: (None, "large-v3", "cpu", "int8"),
        ),
    )

    success = TranscriberGUI._transcribe_batch_media_file(
        fake_gui,
        file_path="recording.wav",
        transcription_config=TranscriptionConfig(batch_backend="compare"),
        grammar_config=GrammarConfig(enabled=False),
        source_label="microphone",
        openai_status="Transcribing microphone recording with OpenAI...",
        openai_progress="Preparing microphone audio for OpenAI transcription...\n",
        whisper_reuse_progress="Reusing the loaded Whisper runtime for microphone transcription...\n",
        cuda_retry_progress_prefix="CUDA ran out of memory during microphone transcription",
    )

    assert success is True
    assert queued_snapshots == [("=== OpenAI ===\nopenai text\n\n=== Local Whisper ===\nlocal text", None)]


def test_record_audio_thread_emits_captured_audio(monkeypatch) -> None:
    """The microphone thread should capture samples and queue them for post-stop transcription."""
    class _FakeCallbackStop(Exception):
        pass

    class _FakeInputStream:
        def __init__(self, *, callback, **kwargs) -> None:
            self.callback = callback

        def __enter__(self):
            self.callback(np.asarray([[0.25], [-0.5]], dtype=np.float32), 2, None, None)
            fake_gui._recording_stop_event.set()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    fake_sounddevice = SimpleNamespace(
        CallbackStop=_FakeCallbackStop,
        PortAudioError=RuntimeError,
        query_devices=lambda *args, **kwargs: [
            {"name": "Mic", "max_input_channels": 1, "default_samplerate": 16000}
        ] if not args else {"name": "Mic", "max_input_channels": 1, "default_samplerate": 16000},
        InputStream=_FakeInputStream,
    )
    fake_gui = cast(
        Any,
        SimpleNamespace(
            output_queue=queue.Queue(),
            _recording_stop_event=threading.Event(),
            _audio_buffer_lock=threading.Lock(),
            _accumulated_audio_buffer=[],
            _recording_sample_rate=16000,
        ),
    )
    monkeypatch.setattr(gt, "_get_sounddevice_module", lambda: fake_sounddevice)

    TranscriberGUI.record_audio_thread(fake_gui, "0: Mic")

    messages = _drain_queue(fake_gui.output_queue)
    captured = [message for message in messages if message[0] == "recording_captured"]
    assert len(captured) == 1
    assert captured[0][1] == [0.25, -0.5]
    assert captured[0][3] == 16000
    assert ("recording_thread_done", "") in messages


def test_check_dependencies_on_startup_runs_off_main_gpu_probe(monkeypatch) -> None:
    """Startup dependency checks should skip the duplicate GPU probe and queue UI work."""
    queued_messages: list[tuple[object, ...]] = []
    detect_calls = {"count": 0}
    dependency_calls: list[bool] = []

    def _detect_gpu_on_startup() -> None:
        detect_calls["count"] += 1

    def _fake_check_dependencies(*, include_gpu_probe: bool = True):
        dependency_calls.append(include_gpu_probe)
        return False, ["FFmpeg"]

    fake_gui = cast(
        Any,
        SimpleNamespace(
            output_queue=SimpleNamespace(put=lambda item: queued_messages.append(item)),
            _detect_gpu_on_startup=_detect_gpu_on_startup,
        ),
    )

    monkeypatch.setattr(gt, "check_dependencies", _fake_check_dependencies)

    TranscriberGUI._check_dependencies_on_startup(fake_gui)

    assert queued_messages == [("startup_missing_dependencies", ("FFmpeg",))]
    assert detect_calls["count"] == 1
    assert dependency_calls == [False]


def test_load_whisper_model_frees_grammar_vram_before_cuda_load(monkeypatch) -> None:
    """Loading Whisper on CUDA should unload grammar models and empty the CUDA cache first."""
    unload_calls: list[str] = []
    gc_calls: list[str] = []
    empty_cache_calls: list[str] = []

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            empty_cache_calls.append("empty")

        @staticmethod
        def get_device_name(index):
            return "RTX Test"

    fake_gui = cast(
        Any,
        SimpleNamespace(
            whisper_model=None,
            _loaded_whisper_model_name=None,
            _loaded_compute_type=None,
            _whisper_device="cpu",
            output_queue=queue.Queue(),
            theme=SimpleNamespace(colors=SimpleNamespace(warning="#f90", success_light="#0f0")),
            config=SimpleNamespace(transcription=TranscriptionConfig()),
        ),
    )

    monkeypatch.setattr(gt, "unload_gector", lambda: unload_calls.append("unload"))
    monkeypatch.setattr(gt.gc, "collect", lambda: gc_calls.append("collect") or 0)
    monkeypatch.setattr(gt, "get_whisper_device_and_compute_type", lambda **kwargs: ("cuda", "float16"))
    monkeypatch.setattr(gt, "get_torch", lambda context: SimpleNamespace(cuda=_FakeCuda()))
    monkeypatch.setattr(
        gt,
        "_load_whisper_pipeline_with_fallback",
        lambda model_name, *, device, compute_type: (None, object(), "cuda", "float16"),
    )

    TranscriberGUI.load_whisper_model(
        fake_gui,
        model_name="large-v3",
        transcription_config=TranscriptionConfig(
            whisper_model="large-v3",
            device_preference="cuda",
            compute_type="float16",
        ),
    )

    progress_messages = [
        cast(str, msg[1]) for msg in _drain_queue(fake_gui.output_queue) if msg[0] == "progress"
    ]

    assert unload_calls == ["unload"]
    assert gc_calls == ["collect"]
    assert empty_cache_calls == ["empty"]
    assert any(
        "Freeing grammar-model GPU memory before loading Whisper" in message
        for message in progress_messages
    )


def test_load_whisper_model_releases_existing_pipeline_before_replacement(monkeypatch) -> None:
    """Switching microphone models should drop the old pipeline before allocating the new one."""
    empty_cache_calls: list[str] = []

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            empty_cache_calls.append("empty")

    fake_gui = cast(
        Any,
        SimpleNamespace(
            whisper_model=object(),
            _loaded_whisper_model_name="large-v3",
            _loaded_compute_type="float16",
            _requested_whisper_device="cuda",
            _requested_compute_type="float16",
            _whisper_device="cuda",
            output_queue=queue.Queue(),
            theme=SimpleNamespace(colors=SimpleNamespace(warning="#f90", success_light="#0f0")),
            config=SimpleNamespace(transcription=TranscriptionConfig()),
        ),
    )

    monkeypatch.setattr(gt, "get_whisper_device_and_compute_type", lambda **kwargs: ("cpu", "int8"))
    monkeypatch.setattr(gt, "get_torch", lambda context: SimpleNamespace(cuda=_FakeCuda()))
    monkeypatch.setattr(gt.gc, "collect", lambda: 0)
    monkeypatch.setattr(gt, "unload_gector", lambda: None)

    def _load_pipeline(model_name, *, device, compute_type):
        assert fake_gui.whisper_model is None
        assert fake_gui._loaded_whisper_model_name is None
        assert device == "cpu"
        assert compute_type == "int8"
        return None, object(), "cpu", "int8"

    monkeypatch.setattr(gt, "_load_whisper_pipeline_with_fallback", _load_pipeline)

    TranscriberGUI.load_whisper_model(
        fake_gui,
        model_name="large-v3",
        transcription_config=TranscriptionConfig(
            whisper_model="large-v3",
            device_preference="cpu",
            compute_type="int8",
        ),
    )

    progress_messages = [
        cast(str, msg[1]) for msg in _drain_queue(fake_gui.output_queue) if msg[0] == "progress"
    ]

    assert empty_cache_calls == ["empty"]
    assert fake_gui.whisper_model is not None
    assert fake_gui._loaded_whisper_model_name == "large-v3"
    assert fake_gui._requested_whisper_device == "cpu"
    assert fake_gui._requested_compute_type == "int8"
    assert fake_gui._whisper_device == "cpu"
    assert fake_gui._loaded_compute_type == "int8"
    assert any("Releasing previously loaded Whisper model" in message for message in progress_messages)


def test_process_queue_recording_reset_clears_timer_and_capture_state() -> None:
    """Recording reset should stop the duration timer and discard stale capture state."""
    record_button_states: list[bool] = []
    timer_stops: list[str] = []
    label_visibility: list[bool] = []
    label_clears: list[str] = []
    record_enabled: list[bool] = []
    mic_enabled: list[bool] = []
    stop_event = threading.Event()

    fake_gui = cast(
        Any,
        SimpleNamespace(
            output_queue=queue.Queue(),
            is_recording=True,
            config=SimpleNamespace(recording=SimpleNamespace(sample_rate=16000)),
            _recording_stop_event=stop_event,
            _recording_start_time=123.0,
            _recording_duration_seconds=14,
            _recording_sample_rate=48000,
            _audio_buffer_lock=threading.Lock(),
            _accumulated_audio_buffer=[0.1, 0.2],
            recording_timer=SimpleNamespace(stop=lambda: timer_stops.append("stop")),
            recording_duration_label=SimpleNamespace(
                clear=lambda: label_clears.append("clear"),
                setVisible=lambda visible: label_visibility.append(visible),
            ),
            record_button=SimpleNamespace(setEnabled=lambda enabled: record_enabled.append(enabled)),
            _set_microphone_selection_enabled=lambda enabled: mic_enabled.append(enabled),
            _set_record_button_state=lambda *, recording: record_button_states.append(recording),
        ),
    )
    fake_gui._reset_recording_ui_state = lambda: TranscriberGUI._reset_recording_ui_state(fake_gui)
    fake_gui.output_queue.put(("recording_reset", ""))

    TranscriberGUI.process_queue(fake_gui)

    assert record_button_states == [False]
    assert stop_event.is_set() is True
    assert fake_gui._recording_start_time is None
    assert fake_gui._recording_duration_seconds == 0
    assert fake_gui._recording_sample_rate == 16000
    assert fake_gui._accumulated_audio_buffer == []
    assert timer_stops == ["stop"]
    assert label_clears == ["clear"]
    assert label_visibility == [False]
    assert record_enabled == [True]
    assert mic_enabled == [True]


def test_process_queue_transcribe_finished_requires_success_flag() -> None:
    """The transcript card should not flash success when the worker only reports completion cleanup."""
    flashed_cards: list[object] = []
    transcript_card = object()

    fake_gui = cast(
        Any,
        SimpleNamespace(
            output_queue=queue.Queue(),
            transcribe_button=SimpleNamespace(setEnabled=lambda enabled: None),
            cancel_youtube_button=SimpleNamespace(
                setVisible=lambda visible: None,
                setEnabled=lambda enabled: None,
            ),
            _transcript_card=transcript_card,
            _flash_card_success=lambda card: flashed_cards.append(card),
        ),
    )
    fake_gui.output_queue.put(("transcribe_finished", False))

    TranscriberGUI.process_queue(fake_gui)

    assert flashed_cards == []


def test_process_queue_local_file_done_flashes_on_explicit_success() -> None:
    """Local-file completion should flash success only when the worker reports success."""
    flashed_cards: list[object] = []
    transcript_card = object()

    fake_gui = cast(
        Any,
        SimpleNamespace(
            output_queue=queue.Queue(),
            transcribe_file_button=SimpleNamespace(setEnabled=lambda enabled: None),
            browse_file_button=SimpleNamespace(setEnabled=lambda enabled: None),
            transcribe_thread=object(),
            _transcript_card=transcript_card,
            _flash_card_success=lambda card: flashed_cards.append(card),
        ),
    )
    fake_gui.output_queue.put(("local_file_done", True))

    TranscriberGUI.process_queue(fake_gui)

    assert flashed_cards == [transcript_card]


def test_detect_gpu_on_startup_queues_unknown_state(monkeypatch) -> None:
    """Background GPU detection should report failures through the queue instead of touching widgets."""
    queued_messages: list[tuple[object, ...]] = []
    fake_gui = cast(
        Any,
        SimpleNamespace(output_queue=SimpleNamespace(put=lambda item: queued_messages.append(item))),
    )

    monkeypatch.setattr(gt, "get_whisper_cuda_status", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    TranscriberGUI._detect_gpu_on_startup(fake_gui)

    assert queued_messages == [("gpu_status_unknown", "boom")]


def test_format_missing_dependencies_message_lists_all_entries() -> None:
    """The startup warning dialog should keep install guidance centralized and readable."""
    message = gt._format_missing_dependencies_message(
        [
            "FFmpeg/ffprobe (required for audio processing)",
            "faster-whisper (required for transcription)",
        ]
    )

    assert "Missing Dependencies Detected" in message
    assert "FFmpeg/ffprobe (required for audio processing)" in message
    assert "faster-whisper (required for transcription)" in message
    assert "pip install -r requirements.txt" in message


def _build_fake_gui() -> Any:
    """Create a lightweight stand-in for TranscriberGUI worker-thread tests."""
    return cast(
        Any,
        SimpleNamespace(
            output_queue=queue.Queue(),
            theme=SimpleNamespace(colors=SimpleNamespace(warning="#f90", success_light="#0f0")),
            _youtube_cancel_event=threading.Event(),
        ),
    )


def _drain_queue(target_queue: queue.Queue[tuple[object, ...]]) -> list[tuple[object, ...]]:
    """Collect all queued GUI messages."""
    messages: list[tuple[object, ...]] = []
    while not target_queue.empty():
        messages.append(target_queue.get_nowait())
    return messages


def test_youtube_thread_does_not_duplicate_snapshot_when_grammar_fails(monkeypatch) -> None:
    """Grammar failures should keep the immediate transcript snapshot single-shot."""
    fake_gui = _build_fake_gui()
    segments = [make_transcript_segment(start=0.0, end=1.0, text="raw transcript")]

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target_queue: (gt.QueueHandler(target_queue), []))
    monkeypatch.setattr(gt, "_detach_worker_queue_logging", lambda queue_handler, logger_states: None)
    monkeypatch.setattr(gt, "validate_youtube_url", lambda url: (True, ""))
    monkeypatch.setattr(gt, "extract_video_id", lambda url: "video123")
    monkeypatch.setattr(gt, "get_youtube_transcript", lambda video_id: ("raw transcript", list(segments)))

    def _raise_grammar_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gt, "post_process_grammar", _raise_grammar_error)

    TranscriberGUI.transcribe_youtube_thread(
        fake_gui,
        "https://youtube.com/watch?v=video123",
        "plain",
        TranscriptionConfig(),
        GrammarConfig(enabled=True),
    )

    messages = _drain_queue(fake_gui.output_queue)
    transcript_messages = [message for message in messages if message[0] == "transcript"]
    segment_messages = [message for message in messages if message[0] == "segments"]

    assert transcript_messages == [("transcript", "raw transcript")]
    assert segment_messages == [("segments", segments)]
    assert any(
        message[0] == "progress" and "Grammar correction skipped: boom" in cast(str, message[1])
        for message in messages
    )


def test_youtube_thread_emits_only_final_snapshot_after_successful_grammar(monkeypatch) -> None:
    """Successful grammar correction should only publish the corrected final snapshot."""
    fake_gui = _build_fake_gui()
    raw_segments = [make_transcript_segment(start=0.0, end=1.0, text="raw transcript")]
    fixed_segments = [make_transcript_segment(start=0.0, end=1.0, text="fixed transcript")]

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target_queue: (gt.QueueHandler(target_queue), []))
    monkeypatch.setattr(gt, "_detach_worker_queue_logging", lambda queue_handler, logger_states: None)
    monkeypatch.setattr(gt, "validate_youtube_url", lambda url: (True, ""))
    monkeypatch.setattr(gt, "extract_video_id", lambda url: "video123")
    monkeypatch.setattr(gt, "get_youtube_transcript", lambda video_id: ("raw transcript", list(raw_segments)))
    monkeypatch.setattr(
        gt,
        "post_process_grammar",
        lambda **kwargs: ("fixed transcript", list(fixed_segments), True),
    )

    TranscriberGUI.transcribe_youtube_thread(
        fake_gui,
        "https://youtube.com/watch?v=video123",
        "plain",
        TranscriptionConfig(),
        GrammarConfig(enabled=True),
    )

    messages = _drain_queue(fake_gui.output_queue)
    assert [message for message in messages if message[0] == "transcript"] == [("transcript", "fixed transcript")]
    assert [message for message in messages if message[0] == "segments"] == [("segments", fixed_segments)]


def test_local_file_thread_does_not_duplicate_snapshot_when_grammar_fails(monkeypatch) -> None:
    """Local-file grammar failures should not resend identical transcript data."""
    fake_gui = _build_fake_gui()
    segments = [make_transcript_segment(start=0.0, end=1.0, text="raw transcript")]

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target_queue: (gt.QueueHandler(target_queue), []))
    monkeypatch.setattr(gt, "_detach_worker_queue_logging", lambda queue_handler, logger_states: None)
    monkeypatch.setattr(gt, "unload_gector", lambda: None)
    monkeypatch.setattr(gt, "get_torch", lambda *args, **kwargs: None)
    monkeypatch.setattr(gt, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        gt,
        "transcribe_local_file",
        lambda **kwargs: ("raw transcript", list(segments)),
    )

    def _raise_grammar_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gt, "post_process_grammar", _raise_grammar_error)

    TranscriberGUI.transcribe_local_file_thread(
        fake_gui,
        "sample.wav",
        TranscriptionConfig(batch_backend="local_whisper"),
        GrammarConfig(enabled=True),
    )

    messages = _drain_queue(fake_gui.output_queue)
    transcript_messages = [message for message in messages if message[0] == "transcript"]
    segment_messages = [message for message in messages if message[0] == "segments"]

    assert transcript_messages == [("transcript", "raw transcript")]
    assert segment_messages == [("segments", segments)]
    assert any(
        message[0] == "progress" and "Grammar correction skipped: boom" in cast(str, message[1])
        for message in messages
    )


def test_local_file_thread_emits_only_final_snapshot_after_successful_grammar(monkeypatch) -> None:
    """Successful local-file grammar correction should only publish the corrected result."""
    fake_gui = _build_fake_gui()
    raw_segments = [make_transcript_segment(start=0.0, end=1.0, text="raw transcript")]
    fixed_segments = [make_transcript_segment(start=0.0, end=1.0, text="fixed transcript")]

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target_queue: (gt.QueueHandler(target_queue), []))
    monkeypatch.setattr(gt, "_detach_worker_queue_logging", lambda queue_handler, logger_states: None)
    monkeypatch.setattr(gt, "unload_gector", lambda: None)
    monkeypatch.setattr(gt, "get_torch", lambda *args, **kwargs: None)
    monkeypatch.setattr(gt, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        gt,
        "transcribe_local_file",
        lambda **kwargs: ("raw transcript", list(raw_segments)),
    )
    monkeypatch.setattr(
        gt,
        "post_process_grammar",
        lambda **kwargs: ("fixed transcript", list(fixed_segments), True),
    )

    TranscriberGUI.transcribe_local_file_thread(
        fake_gui,
        "sample.wav",
        TranscriptionConfig(batch_backend="local_whisper"),
        GrammarConfig(enabled=True),
    )

    messages = _drain_queue(fake_gui.output_queue)
    assert [message for message in messages if message[0] == "transcript"] == [("transcript", "fixed transcript")]
    assert [message for message in messages if message[0] == "segments"] == [("segments", fixed_segments)]


def test_youtube_compare_mode_clears_mismatched_timestamp_segments(monkeypatch, tmp_path) -> None:
    """Combined compare transcripts should not publish one backend's timestamps as if they covered both sections."""
    fake_gui = _build_fake_gui()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    openai_segments = [make_transcript_segment(start=0.0, end=1.0, text="openai text")]
    local_segments = [make_transcript_segment(start=0.0, end=1.0, text="local text")]

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target_queue: (gt.QueueHandler(target_queue), []))
    monkeypatch.setattr(gt, "_detach_worker_queue_logging", lambda queue_handler, logger_states: None)
    monkeypatch.setattr(gt, "validate_youtube_url", lambda url: (True, ""))
    monkeypatch.setattr(gt, "extract_video_id", lambda url: "video123")
    monkeypatch.setattr(gt, "get_youtube_transcript", lambda video_id: (None, None))
    monkeypatch.setattr(gt, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(gt, "download_audio", lambda url, output_name, ffmpeg_location: str(audio_path))
    monkeypatch.setattr(gt, "is_openai_api_configured", lambda: True)
    monkeypatch.setattr(gt, "transcribe_audio_openai", lambda *args, **kwargs: ("openai text", list(openai_segments)))
    monkeypatch.setattr(gt, "unload_gector", lambda: None)
    monkeypatch.setattr(gt, "get_torch", lambda *args, **kwargs: None)
    monkeypatch.setattr(gt, "transcribe_audio", lambda *args, **kwargs: ("local text", list(local_segments)))

    TranscriberGUI.transcribe_youtube_thread(
        fake_gui,
        "https://youtube.com/watch?v=video123",
        "plain",
        TranscriptionConfig(batch_backend="compare"),
        GrammarConfig(enabled=False),
    )

    messages = _drain_queue(fake_gui.output_queue)

    assert [message for message in messages if message[0] == "segments"] == [("segments", [])]
    assert [message for message in messages if message[0] == "transcript"] == [
        ("transcript", "=== OpenAI ===\nopenai text\n\n=== Local Whisper ===\nlocal text")
    ]


def test_local_file_compare_mode_clears_mismatched_timestamp_segments(monkeypatch) -> None:
    """Local-file compare output should not reuse timestamps from only the OpenAI half."""
    fake_gui = _build_fake_gui()
    openai_segments = [make_transcript_segment(start=0.0, end=1.0, text="openai text")]
    local_segments = [make_transcript_segment(start=0.0, end=1.0, text="local text")]

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target_queue: (gt.QueueHandler(target_queue), []))
    monkeypatch.setattr(gt, "_detach_worker_queue_logging", lambda queue_handler, logger_states: None)
    monkeypatch.setattr(gt, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(gt, "is_openai_api_configured", lambda: True)
    monkeypatch.setattr(gt, "transcribe_local_file_openai", lambda *args, **kwargs: ("openai text", list(openai_segments)))
    monkeypatch.setattr(gt, "unload_gector", lambda: None)
    monkeypatch.setattr(gt, "get_torch", lambda *args, **kwargs: None)
    monkeypatch.setattr(gt, "transcribe_local_file", lambda **kwargs: ("local text", list(local_segments)))

    TranscriberGUI.transcribe_local_file_thread(
        fake_gui,
        "sample.wav",
        TranscriptionConfig(batch_backend="compare"),
        GrammarConfig(enabled=False),
    )

    messages = _drain_queue(fake_gui.output_queue)

    assert [message for message in messages if message[0] == "segments"] == [("segments", [])]
    assert [message for message in messages if message[0] == "transcript"] == [
        ("transcript", "=== OpenAI ===\nopenai text\n\n=== Local Whisper ===\nlocal text")
    ]


def test_local_file_thread_retries_full_job_on_cpu_after_cuda_oom(monkeypatch) -> None:
    """Recoverable CUDA OOMs should trigger a whole-job CPU retry instead of a hard failure."""
    fake_gui = _build_fake_gui()
    fake_gui._build_cpu_recovery_transcription_config = (
        lambda config: TranscriberGUI._build_cpu_recovery_transcription_config(fake_gui, config)
    )
    recovered_segments = [make_transcript_segment(start=0.0, end=1.0, text="recovered transcript")]
    seen_configs: list[tuple[str, str, int]] = []

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target_queue: (gt.QueueHandler(target_queue), []))
    monkeypatch.setattr(gt, "_detach_worker_queue_logging", lambda queue_handler, logger_states: None)
    monkeypatch.setattr(gt, "unload_gector", lambda: None)
    monkeypatch.setattr(gt, "get_torch", lambda *args, **kwargs: None)
    monkeypatch.setattr(gt, "find_ffmpeg", lambda: "ffmpeg")

    def _transcribe_local_file(**kwargs):
        config = kwargs["config"]
        seen_configs.append((config.device_preference, config.compute_type, config.batch_size))
        if len(seen_configs) == 1:
            raise RuntimeError("CUDA failed with error out of memory")
        return "recovered transcript", list(recovered_segments)

    monkeypatch.setattr(gt, "transcribe_local_file", _transcribe_local_file)

    TranscriberGUI.transcribe_local_file_thread(
        fake_gui,
        "sample.wav",
        TranscriptionConfig(
            batch_backend="local_whisper",
            device_preference="auto",
            compute_type="float16",
            batch_size=32,
            cpu_fallback_batch_size=8,
        ),
        GrammarConfig(enabled=False),
    )

    messages = _drain_queue(fake_gui.output_queue)

    assert seen_configs == [("auto", "float16", 32), ("cpu", "int8", 8)]
    assert ("transcript", "recovered transcript") in messages
    assert ("segments", recovered_segments) in messages
    assert ("transcribe_finished", True) in messages
    assert ("local_file_done", True) in messages
    assert any(
        message[0] == "progress" and "retrying full job on CPU" in cast(str, message[1])
        for message in messages
    )
    assert not any(message[0] == "error" for message in messages)


def test_local_file_thread_reuses_loaded_whisper_runtime(monkeypatch) -> None:
    """Local-file transcription should reuse an already-loaded compatible Whisper runtime."""
    fake_gui = _build_fake_gui()
    fake_gui.whisper_model = object()
    fake_gui._loaded_whisper_model_name = "large-v3"
    fake_gui._loaded_compute_type = "float16"
    fake_gui._requested_whisper_device = "cuda"
    fake_gui._requested_compute_type = "float16"
    fake_gui._whisper_device = "cuda"

    seen_execution_states: list[tuple[str, str, str]] = []

    monkeypatch.setattr(gt, "_attach_worker_queue_logging", lambda target_queue: (gt.QueueHandler(target_queue), []))
    monkeypatch.setattr(gt, "_detach_worker_queue_logging", lambda queue_handler, logger_states: None)
    monkeypatch.setattr(gt, "unload_gector", lambda: None)
    monkeypatch.setattr(gt, "get_torch", lambda *args, **kwargs: None)
    monkeypatch.setattr(gt, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(gt, "get_whisper_device_and_compute_type", lambda **kwargs: ("cuda", "float16"))

    def _transcribe_local_file(**kwargs):
        execution_state = kwargs["execution_state"]
        assert execution_state is not None
        seen_execution_states.append(
            (execution_state.model_name, execution_state.device, execution_state.compute_type)
        )
        observer = kwargs.get("execution_state_observer")
        if observer is not None:
            observer(execution_state)
        return "reused transcript", [make_transcript_segment(start=0.0, end=1.0, text="reused transcript")]

    monkeypatch.setattr(gt, "transcribe_local_file", _transcribe_local_file)

    TranscriberGUI.transcribe_local_file_thread(
        fake_gui,
        "sample.wav",
        TranscriptionConfig(batch_backend="local_whisper", whisper_model="large-v3"),
        GrammarConfig(enabled=False),
    )

    messages = _drain_queue(fake_gui.output_queue)
    assert seen_execution_states == [("large-v3", "cuda", "float16")]
    assert ("transcript", "reused transcript") in messages
    assert any(
        message[0] == "progress" and "Reusing the loaded Whisper runtime" in cast(str, message[1])
        for message in messages
    )
