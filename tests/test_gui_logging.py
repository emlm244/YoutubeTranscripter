"""Regression tests for the GUI logging bridge."""

from __future__ import annotations

import logging
import queue
import threading
from types import SimpleNamespace
from typing import Any, cast

import gui_transcriber as gt

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


def test_build_transcription_config_keeps_audio_cleanup_toggles_independent(monkeypatch) -> None:
    """Noise reduction and normalization should remain independently configurable."""
    fake_gui = cast(
        Any,
        SimpleNamespace(
            config=SimpleNamespace(transcription=TranscriptionConfig()),
            preset_combo=SimpleNamespace(currentData=lambda: gt.DEFAULT_PRESET),
            hotwords_input=SimpleNamespace(text=lambda: ""),
            filler_cleanup_checkbox=SimpleNamespace(isChecked=lambda: True),
            noise_reduction_checkbox=SimpleNamespace(isChecked=lambda: False),
            normalize_audio_checkbox=SimpleNamespace(isChecked=lambda: True),
        ),
    )
    monkeypatch.setattr(gt, "apply_preset", lambda config, preset_name: None)

    config = TranscriberGUI._build_transcription_config(fake_gui)

    assert config.noise_reduction_enabled is False
    assert config.normalize_audio is True


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
        TranscriptionConfig(),
        GrammarConfig(enabled=True),
    )

    messages = _drain_queue(fake_gui.output_queue)
    assert [message for message in messages if message[0] == "transcript"] == [("transcript", "fixed transcript")]
    assert [message for message in messages if message[0] == "segments"] == [("segments", fixed_segments)]
