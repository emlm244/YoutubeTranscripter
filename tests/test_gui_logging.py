"""Regression tests for the GUI logging bridge."""

from __future__ import annotations

import logging
import queue

from gui_transcriber import (
    QueueLogger,
    _attach_worker_queue_logging,
    _detach_worker_queue_logging,
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
