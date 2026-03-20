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
    TranscriberGUI,
    _attach_worker_queue_logging,
    _detach_worker_queue_logging,
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
        assert target_queue.get_nowait() == ("progress", "Backend progress is visible")
    finally:
        _detach_worker_queue_logging(queue_handler, logger_states)

    for logger_name in logger_names:
        logger_obj = logging.getLogger(logger_name)
        assert queue_handler not in logger_obj.handlers
        assert logger_obj.level == original_levels[logger_name]


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
