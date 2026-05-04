"""
Speech Transcriber - Modern GUI Application with PyQt6.

Provides YouTube transcription and high-quality microphone recording with batch
speech-to-text processing using OpenAI or local Whisper.

Features Material Design styling, responsive layout, and theme system.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import json
import logging
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import warnings
from io import StringIO
from logging.handlers import RotatingFileHandler
from typing import Any, Optional, cast

import gui_runtime_bootstrap  # noqa: F401
from app_paths import get_log_path
from torch_runtime import get_torch

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

# WhisperModel imported lazily in load_whisper_model() to speed up GUI startup

from audio_preprocessor import preprocess_array
from youtube_transcriber import (
    _WhisperExecutionState,
    _build_runtime_transcribe_kwargs,
    _load_whisper_pipeline_with_fallback,
    _needs_cuda_cpu_fallback,
    _normalize_transcript_segments,
    build_suspicion_retry_kwargs,
    check_dependencies,
    download_audio,
    extract_video_id,
    find_ffmpeg,
    format_srt_timestamp,
    format_transcript_with_timestamps,
    format_transcript_as_srt,
    format_transcript_as_json,
    get_youtube_transcript,
    is_openai_api_configured,
    transcribe_audio,
    transcribe_audio_openai,
    transcribe_local_file,
    transcribe_local_file_openai,
    transcription_result_looks_suspicious,
    validate_youtube_url,
    get_whisper_cuda_status,
    get_whisper_device_and_compute_type,
)

# Import new modules
from config import (
    ACCURACY_PRESETS,
    BATCH_BACKEND_OPTIONS,
    COMPUTE_TYPE_OPTIONS,
    DEFAULT_PRESET,
    DEVICE_PREFERENCE_OPTIONS,
    GRAMMAR_BACKEND_OPTIONS,
    GrammarConfig,
    OPENAI_BATCH_MODEL_OPTIONS,
    TranscriptionConfig,
    WHISPER_MODEL_OPTIONS,
    apply_preset,
    get_config,
    save_config,
)
from grammar_postprocessor import check_grammar_status, post_process_grammar, unload_gector
from openai_realtime import (
    OPENAI_REALTIME_INPUT_SAMPLE_RATE,
    OpenAIRealtimeTranscriptionSession,
)
from themes import get_theme_manager
from transcript_types import TranscriptSegments, make_transcript_segment
from widgets import (
    MaterialCard,
    GlassCard,
    MaterialButton,
    ResponsiveSplitter,
)

# Suppress ctranslate2 pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
# Suppress PyTorch CUDA compatibility warning for RTX 5080 (sm_120 Blackwell)
warnings.filterwarnings("ignore", message=".*CUDA capability sm_120.*", category=UserWarning)

OPENAI_REALTIME_FINALIZATION_TIMEOUT_S = 10.0
OPENAI_REALTIME_FINALIZATION_POLL_S = 0.1


TRANSCRIPTION_SETTING_HELP_TEXTS: dict[str, str] = {
    "Preset:": (
        "Loads a recommended bundle of runtime settings.\n"
        "Speed favors turnaround time, Balanced is the middle ground, and Maximum Accuracy uses heavier decoding.\n"
        "You can still tweak any setting afterward."
    ),
    "Model:": (
        "Uses Whisper Large v3 for local transcription.\n"
        "This is the single supported local Whisper model for this app."
    ),
    "Backend:": (
        "Choose the batch transcription engine for YouTube fallback and local files.\n"
        "OpenAI uses the cloud API, Local Whisper uses the installed model, and Compare runs both."
    ),
    "OpenAI Model:": (
        "Choose the OpenAI model for batch audio/video transcription.\n"
        "This requires OPENAI_API_KEY in the environment."
    ),
    "Language:": (
        "Optionally force a language code such as en, en-US, es, or fr.\n"
        "Leave this blank to let Whisper auto-detect the language."
    ),
    "Hotwords:": (
        "Recognition hints for names, acronyms, and jargon.\n"
        "Separate terms with commas.\n"
        "These bias recognition but do not force words into the transcript."
    ),
    "Grammar Fix": (
        "Runs local grammar and spelling cleanup after transcription.\n"
        "This improves readability, but it is less raw/verbatim than leaving the transcript untouched."
    ),
    "Clean Fillers": (
        "Removes filler words such as um, uh, you know, and similar speech clutter.\n"
        "Leave this off if you want the most faithful verbatim transcript."
    ),
    "Noise Reduction": (
        "Applies spectral denoising before Whisper runs.\n"
        "Useful for hiss, fans, HVAC, or steady room noise."
    ),
    "Normalize Audio": (
        "Applies loudness normalization before Whisper runs.\n"
        "Useful when speech is too quiet or inconsistent."
    ),
    "Filter Hallucinations": (
        "Removes common silence, music, and low-signal junk phrases Whisper can invent.\n"
        "This is one of the main transcript-safety guardrails."
    ),
    "Deduplicate Loops": (
        "Collapses obvious repeated decode loops before the transcript is shown.\n"
        "Useful when the model gets stuck repeating the same phrase."
    ),
    "Word Timestamps": (
        "Keeps word-level timing when the backend provides it.\n"
        "Useful for precise timestamps and subtitle work."
    ),
    "Context Carryover": (
        "Lets earlier decoded text influence later segments.\n"
        "This can improve continuity, but it can also carry mistakes forward."
    ),
    "Use VAD": (
        "Uses voice activity detection to isolate speech before Whisper decodes.\n"
        "Usually helpful for meetings and mixed-noise recordings."
    ),
    "Device:": (
        "Select the preferred runtime device.\n"
        "Auto chooses GPU when available, Prefer GPU pushes for CUDA, and Force CPU disables GPU use."
    ),
    "Compute:": (
        "Select the model compute type.\n"
        "Auto picks float16 on GPU and int8 on CPU.\n"
        "Lower-precision modes can reduce memory usage."
    ),
    "Batch:": (
        "Controls how many chunks Whisper processes together.\n"
        "Higher values are usually faster, but they need more VRAM or RAM."
    ),
    "CPU Fallback Batch:": (
        "Backup batch size used if the transcription falls back to CPU.\n"
        "Keep this lower than the main batch size to avoid CPU overload."
    ),
    "Beam:": (
        "Beam search width.\n"
        "Higher values can improve accuracy, but they slow decoding."
    ),
    "Temperature:": (
        "Decoding randomness.\n"
        "Lower values are more deterministic; 0.0 is the most stable."
    ),
    "GPU Memory %:": (
        "Best-effort PyTorch VRAM budget for torch-based components.\n"
        "Higher values allow more headroom for GPU work, but this is not a strict hard cap for every backend."
    ),
    "No-Speech Threshold:": (
        "How easily Whisper decides that a chunk is silence.\n"
        "Lower values make it more willing to transcribe faint speech."
    ),
    "VAD Threshold:": (
        "How strict the speech detector is.\n"
        "Higher values are stricter and may ignore quieter speech."
    ),
    "Min Speech (ms):": (
        "Minimum speech fragment length the VAD will keep.\n"
        "Shorter fragments than this are treated as noise."
    ),
    "Min Silence (ms):": (
        "How much silence is required before VAD splits speech into a new chunk."
    ),
    "Speech Pad (ms):": (
        "Extra audio kept before and after each detected speech region.\n"
        "This helps prevent clipped word starts or endings."
    ),
    "Repetition Penalty:": (
        "Discourages the decoder from repeating the same text over and over.\n"
        "Higher values are more aggressive."
    ),
    "No-Repeat N-gram:": (
        "Blocks repeated word groups of the given size.\n"
        "Use 0 to effectively disable this guardrail."
    ),
    "Patience:": (
        "How long beam search keeps exploring alternate candidates.\n"
        "Higher values can help difficult audio, but they are slower."
    ),
    "Length Penalty:": (
        "Bias toward shorter or longer outputs.\n"
        "1.0 is neutral."
    ),
    "Hallucination Silence:": (
        "Extra silence-based safeguard used during decoding to reduce hallucinated output."
    ),
    "Grammar Backend:": (
        "Choose which grammar engine to use.\n"
        "Auto prefers GECToR and falls back to LanguageTool when needed."
    ),
    "Grammar Language:": (
        "Locale used for grammar rules, such as en-US."
    ),
    "GECToR Batch:": (
        "Batch size for the grammar correction model.\n"
        "Higher values are faster on strong GPUs, but use more memory."
    ),
    "GECToR Iterations:": (
        "How many correction passes GECToR makes.\n"
        "More passes can clean more text, but they can also over-edit."
    ),
}


QueueMessage = tuple[Any, ...]
LoggerState = tuple[logging.Logger, int]

BACKEND_LOGGER_LEVELS: dict[str, int] = {
    "youtube_transcriber": logging.INFO,
    "grammar_postprocessor": logging.INFO,
    "faster_whisper": logging.INFO,
}
BACKEND_LOG_NAME = "youtube_transcriber.log"

_backend_file_handler: RotatingFileHandler | None = None
_backend_file_handler_lock = threading.Lock()
WHISPER_SAMPLE_RATE = 16_000


def _get_sounddevice_module():
    """Import sounddevice only when microphone paths need it."""
    import sounddevice

    return sounddevice


def _resample_audio_for_whisper(
    audio: np.ndarray,
    *,
    source_rate: int,
    target_rate: int = WHISPER_SAMPLE_RATE,
) -> np.ndarray:
    """Resample recorded audio to Whisper's expected sample rate."""
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    if audio.size == 1:
        return np.repeat(audio.astype(np.float32, copy=False), max(target_rate // source_rate, 1))

    target_length = max(round(audio.size * (target_rate / source_rate)), 1)
    if target_length == audio.size:
        return audio.astype(np.float32, copy=False)

    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0, audio.size - 1, num=target_length)
    resampled = np.interp(target_positions, source_positions, audio.astype(np.float64, copy=False))
    return resampled.astype(np.float32, copy=False)


@dataclass(frozen=True)
class GrammarPassResult:
    """Outcome of an optional grammar-processing pass."""

    transcript: str
    segments_data: TranscriptSegments
    grammar_enhanced: bool
    completed: bool


def _format_missing_dependencies_message(missing: Sequence[str]) -> str:
    """Build the user-facing startup dependency warning dialog."""
    missing_list = "\n".join(f"  - {name}" for name in missing)
    return (
        "Missing Dependencies Detected:\n\n"
        f"{missing_list}\n\n"
        "Some features may not work correctly.\n\n"
        "To install missing dependencies:\n"
        "1. Open a terminal/command prompt\n"
        "2. Run: pip install -r requirements.txt\n"
        "3. For FFmpeg: winget install FFmpeg (Windows)"
    )


class QueueLogger(StringIO):
    """Redirect stdout writes into the GUI event queue."""

    def __init__(self, target_queue: queue.Queue[QueueMessage]):
        super().__init__()
        self._target_queue = target_queue
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        self._emit_complete_lines()
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            chunk = self._normalize_progress_chunk(self._buffer, complete=False)
            if chunk:
                self._target_queue.put(("progress", chunk))
            self._buffer = ""

    def _emit_complete_lines(self) -> None:
        self._buffer = self._buffer.lstrip("\r")
        lines = self._buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._buffer = lines.pop()
        else:
            self._buffer = ""

        for chunk in lines:
            normalized = self._normalize_progress_chunk(chunk, complete=True)
            if normalized:
                self._target_queue.put(("progress", normalized))

    @staticmethod
    def _normalize_progress_chunk(text: str, *, complete: bool) -> str:
        """Normalize carriage-return progress updates for append-only GUI output."""
        normalized = text.lstrip("\r")
        if complete:
            if normalized.endswith("\r\n"):
                normalized = normalized[:-2] + "\n"
            elif normalized.endswith("\r"):
                normalized = normalized[:-1] + "\n"
        return normalized


class QueueHandler(logging.Handler):
    """Custom logging handler that sends log messages to the GUI queue."""

    def __init__(self, target_queue: queue.Queue[QueueMessage]):
        super().__init__()
        self._target_queue = target_queue
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = QueueLogger._normalize_progress_chunk(self.format(record), complete=True)
            if msg and not msg.endswith("\n"):
                msg += "\n"
            if msg:
                self._target_queue.put(("progress", msg))
        except Exception:
            self.handleError(record)


# Configure GUI logger
gui_logger = logging.getLogger(__name__)


def _build_rotating_file_handler(log_name: str) -> RotatingFileHandler:
    """Create a consistent rotating file handler for GUI/backend logs."""
    file_handler = RotatingFileHandler(
        get_log_path(log_name),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return file_handler


def _get_backend_file_handler() -> RotatingFileHandler:
    """Reuse one backend file handler across the backend logger fan-out."""
    global _backend_file_handler
    if _backend_file_handler is None:
        with _backend_file_handler_lock:
            if _backend_file_handler is None:
                _backend_file_handler = _build_rotating_file_handler(BACKEND_LOG_NAME)
    return _backend_file_handler


def _logger_has_handler_for_path(logger_obj: logging.Logger, log_path: Path) -> bool:
    """Return whether the logger already writes to the given file path."""
    target = str(log_path.resolve())
    for handler in logger_obj.handlers:
        base_filename = getattr(handler, "baseFilename", None)
        if not base_filename:
            continue
        try:
            if str(Path(base_filename).resolve()) == target:
                return True
        except OSError:
            if base_filename == target:
                return True
    return False


def setup_backend_logging_for_gui() -> None:
    """Ensure backend modules write detailed logs when launched from the GUI."""
    backend_log_path = get_log_path(BACKEND_LOG_NAME)
    shared_handler = _get_backend_file_handler()
    for logger_name, level in BACKEND_LOGGER_LEVELS.items():
        logger_obj = logging.getLogger(logger_name)
        logger_obj.setLevel(level)
        if not _logger_has_handler_for_path(logger_obj, backend_log_path):
            logger_obj.addHandler(shared_handler)

    logging.getLogger("yt_dlp").setLevel(logging.WARNING)


def _attach_worker_queue_logging(target_queue: queue.Queue[QueueMessage]) -> tuple[QueueHandler, list[LoggerState]]:
    """Attach GUI progress handlers to the backend loggers used by worker threads."""
    queue_handler = QueueHandler(target_queue)
    queue_handler.setLevel(logging.INFO)

    logger_states: list[LoggerState] = []
    for logger_name, level in BACKEND_LOGGER_LEVELS.items():
        logger_obj = logging.getLogger(logger_name)
        logger_states.append((logger_obj, logger_obj.level))
        logger_obj.setLevel(level)
        logger_obj.addHandler(queue_handler)

    return queue_handler, logger_states


def _detach_worker_queue_logging(queue_handler: QueueHandler, logger_states: list[LoggerState]) -> None:
    """Restore worker logger configuration after a background task completes."""
    for logger_obj, original_level in logger_states:
        logger_obj.removeHandler(queue_handler)
        logger_obj.setLevel(original_level)


def setup_gui_logging(target_queue: queue.Queue[QueueMessage]) -> None:
    """Set up logging for the GUI application."""
    file_handler = _build_rotating_file_handler("gui_transcriber.log")

    queue_handler = QueueHandler(target_queue)
    queue_handler.setLevel(logging.INFO)

    gui_logger.setLevel(logging.DEBUG)
    gui_logger.addHandler(file_handler)
    gui_logger.addHandler(queue_handler)

    logging.getLogger('PyQt6').setLevel(logging.WARNING)


@contextmanager
def _worker_queue_bridge(target_queue: queue.Queue[QueueMessage]):
    """Mirror backend stdout/logging into the GUI queue for the duration of a worker job."""
    queue_handler, logger_states = _attach_worker_queue_logging(target_queue)
    progress_redirector = QueueLogger(target_queue)
    old_stdout = sys.stdout
    try:
        sys.stdout = progress_redirector
        yield
    finally:
        progress_redirector.flush()
        sys.stdout = old_stdout
        _detach_worker_queue_logging(queue_handler, logger_states)


def _queue_transcript_snapshot(
    target_queue: queue.Queue[QueueMessage],
    transcript: str,
    segments_data: TranscriptSegments | None,
    *,
    output_format: str = "plain",
    append: bool = False,
) -> None:
    """Queue transcript text plus timestamp data for the UI thread."""
    normalized_segments = segments_data if segments_data else []
    rendered_transcript = transcript
    if output_format == "timestamped" and normalized_segments:
        rendered_transcript = format_transcript_with_timestamps(normalized_segments)

    target_queue.put(("segments", normalized_segments))
    message_kind = "append_transcript" if append else "transcript"
    target_queue.put((message_kind, rendered_transcript))


def _format_backend_comparison(
    *,
    openai_transcript: str | None,
    local_transcript: str | None,
) -> str:
    sections: list[str] = []
    if openai_transcript is not None:
        sections.append("=== OpenAI ===\n" + (openai_transcript.strip() or "[No speech detected]"))
    if local_transcript is not None:
        sections.append("=== Local Whisper ===\n" + (local_transcript.strip() or "[No speech detected]"))
    return "\n\n".join(sections)


def _apply_optional_grammar_corrections(
    target_queue: queue.Queue[QueueMessage],
    transcript: str,
    segments_data: TranscriptSegments,
    grammar_config: GrammarConfig,
    *,
    warning_color: str | None = None,
    start_status: str | None = None,
    start_progress: str = "Fixing grammar...\n",
    success_progress: str = "Grammar correction applied.\n",
    no_change_progress: str = "Grammar: No changes needed or unavailable.\n",
) -> GrammarPassResult:
    """Run grammar correction when enabled and report the result back to the queue."""
    if not grammar_config.enabled:
        return GrammarPassResult(transcript, segments_data, False, False)

    if start_status:
        target_queue.put(("status", start_status, warning_color))
    target_queue.put(("progress", start_progress))

    try:
        processed_transcript, processed_segments, grammar_enhanced = post_process_grammar(
            text=transcript,
            segments_data=segments_data,
            config=grammar_config,
        )
    except Exception as grammar_exc:
        gui_logger.warning(f"Grammar correction failed: {grammar_exc}")
        target_queue.put(("progress", f"Grammar correction skipped: {grammar_exc}\n"))
        return GrammarPassResult(transcript, segments_data, False, False)

    if processed_segments is None:
        processed_segments = segments_data

    target_queue.put(
        ("progress", success_progress if grammar_enhanced else no_change_progress)
    )
    return GrammarPassResult(processed_transcript, processed_segments, grammar_enhanced, True)


def _build_transcription_complete_status(
    *,
    grammar_enhanced: bool,
    segment_count: int | None = None,
    word_count: int | None = None,
) -> str:
    """Build the success status text shown after a transcription finishes."""
    if word_count is not None:
        status_msg = f"Transcription complete ({word_count} words)"
        if grammar_enhanced:
            status_msg += " - grammar corrected"
        return status_msg

    status_msg = "Transcription complete!"
    if segment_count is not None:
        status_msg += f" ({segment_count} segments)"
    if grammar_enhanced:
        status_msg += " (grammar corrected)"
    return status_msg


class TranscriberGUI(QtWidgets.QMainWindow):
    """PyQt6-based GUI for transcription and OpenAI realtime microphone capture."""

    def __init__(self) -> None:
        super().__init__()

        # Initialize theme manager
        self.theme = get_theme_manager()
        self.config = get_config()
        self._whisper_device = "cpu"

        self.setWindowTitle("Speech Transcriber")
        self._configure_window_size()

        self.output_queue: queue.Queue[QueueMessage] = queue.Queue()
        self.is_recording = False
        self.whisper_model: Optional[Any] = None  # WhisperModel loaded lazily
        self._loaded_whisper_model_name: Optional[str] = None
        self._loaded_compute_type: Optional[str] = None
        self._requested_whisper_device: Optional[str] = None
        self._requested_compute_type: Optional[str] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.transcribe_thread: Optional[threading.Thread] = None
        self._loading_model = False
        self._pending_record_start = False

        # Cancellation support
        self._youtube_cancel_event = threading.Event()
        self._recording_stop_event = threading.Event()

        # Recording timer
        self._recording_start_time: Optional[float] = None
        self._recording_duration_seconds = 0
        self._recording_sample_rate = self.config.recording.sample_rate
        self._accumulated_audio_buffer: list[float] = []
        self._audio_buffer_lock = threading.Lock()

        # Store transcript segments with timestamps (always dict format with 'start', 'end', 'text')
        self._current_segments_data: Optional[TranscriptSegments] = None

        # Set up logging
        setup_gui_logging(self.output_queue)
        setup_backend_logging_for_gui()

        # Apply theme
        app = QtWidgets.QApplication.instance()
        if app:
            self.theme.apply_to_app(cast(QtWidgets.QApplication, app))

        self._build_ui()
        self._connect_signals()

        # Queue processing timer
        self.queue_timer = QtCore.QTimer(self)
        self.queue_timer.timeout.connect(self.process_queue)
        self.queue_timer.start(120)

        # Recording duration timer
        self.recording_timer = QtCore.QTimer(self)
        self.recording_timer.timeout.connect(self._update_recording_duration)
        self.recording_timer.setInterval(1000)

        self.update_status("Ready")
        gui_logger.info("Speech Transcriber GUI initialized")

        self._load_settings()
        self._schedule_startup_tasks()

    def _configure_window_size(self) -> None:
        """Set dynamic window size based on available screen space."""
        screen = QtWidgets.QApplication.primaryScreen()

        if screen:
            available = screen.availableGeometry()
            default_width = min(int(available.width() * 0.75), 1600)
            default_height = min(int(available.height() * 0.80), 1000)
            self.resize(default_width, default_height)

            # Center on screen
            x = (available.width() - default_width) // 2
            y = (available.height() - default_height) // 2
            self.move(x, y)
        else:
            self.resize(1280, 900)

        self.setMinimumSize(800, 600)

    def _set_combo_data(self, combo: QtWidgets.QComboBox, value: str, *, custom_prefix: str = "Custom") -> None:
        """Select combo data, adding a one-off custom item when needed."""
        index = combo.findData(value)
        if index < 0:
            combo.addItem(f"{custom_prefix}: {value}", value)
            index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _create_help_badge(self, tooltip: str) -> QtWidgets.QToolButton:
        """Create the small ? badge used beside setting titles."""
        badge = QtWidgets.QToolButton()
        badge.setText("?")
        badge.setToolTip(tooltip)
        badge.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        badge.setAutoRaise(True)
        badge.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        badge.setFixedSize(18, 18)
        badge.setStyleSheet(
            f"""
            QToolButton {{
                border: 1px solid {self.theme.colors.text_secondary};
                border-radius: 9px;
                color: {self.theme.colors.info};
                background: transparent;
                font-size: 11px;
                font-weight: 700;
                padding: 0px;
            }}
            QToolButton:hover {{
                border-color: {self.theme.colors.info};
                background: rgba(255, 255, 255, 0.06);
            }}
            """
        )
        return badge

    def _create_setting_label_with_help(self, label_text: str, tooltip: str) -> QtWidgets.QWidget:
        """Create a setting title label with a hoverable ? badge."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QtWidgets.QLabel(label_text)
        label.setToolTip(tooltip)

        layout.addWidget(label)
        layout.addWidget(self._create_help_badge(tooltip))
        return container

    def _wrap_setting_widget_with_help(self, widget: QtWidgets.QWidget, tooltip: str) -> QtWidgets.QWidget:
        """Wrap a checkbox-like control with a hoverable ? badge."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        widget.setToolTip(tooltip)
        layout.addWidget(widget)
        layout.addWidget(self._create_help_badge(tooltip))
        return container

    def _sync_runtime_preferences_from_controls(self) -> None:
        """Sync shared runtime-only settings that backend helpers read globally."""
        if hasattr(self, "gpu_memory_spin"):
            self.config.gpu_memory_fraction = self.gpu_memory_spin.value()

    def _build_cpu_recovery_transcription_config(
        self,
        config: TranscriptionConfig,
    ) -> TranscriptionConfig:
        """Clone transcription config for a full-job CPU recovery attempt."""
        cpu_config = TranscriptionConfig(**asdict(config))
        cpu_config.device_preference = "cpu"
        cpu_config.compute_type = "int8"
        cpu_config.batch_size = min(cpu_config.batch_size, cpu_config.cpu_fallback_batch_size)
        return cpu_config

    def _apply_transcription_controls(self, config: TranscriptionConfig) -> None:
        """Push a transcription config into the UI controls."""
        self._set_combo_data(self.batch_backend_combo, config.batch_backend)
        self._set_combo_data(self.openai_batch_model_combo, config.openai_batch_model, custom_prefix="OpenAI")
        self._set_combo_data(self.whisper_model_combo, config.whisper_model, custom_prefix="Model")
        self.language_input.setText(config.language or "")
        self.hotwords_input.setText(config.hotwords or "")
        self.filler_cleanup_checkbox.setChecked(config.clean_filler_words)
        self.noise_reduction_checkbox.setChecked(config.noise_reduction_enabled)
        self.normalize_audio_checkbox.setChecked(config.normalize_audio)
        self.filter_hallucinations_checkbox.setChecked(config.filter_hallucinations)
        self.deduplicate_checkbox.setChecked(config.deduplicate_repeated_segments)
        self.word_timestamps_checkbox.setChecked(config.word_timestamps)
        self.previous_text_checkbox.setChecked(config.condition_on_previous_text)
        self.vad_filter_checkbox.setChecked(config.vad_filter)
        self._set_combo_data(self.device_preference_combo, config.device_preference)
        self._set_combo_data(self.compute_type_combo, config.compute_type)
        self.batch_size_spin.setValue(config.batch_size)
        self.cpu_fallback_batch_spin.setValue(config.cpu_fallback_batch_size)
        self.beam_size_spin.setValue(config.beam_size)
        self.temperature_spin.setValue(config.temperature)
        self.no_speech_threshold_spin.setValue(config.no_speech_threshold)
        self.vad_threshold_spin.setValue(config.vad_threshold)
        self.min_speech_duration_spin.setValue(config.min_speech_duration_ms)
        self.min_silence_duration_spin.setValue(config.min_silence_duration_ms)
        self.speech_pad_spin.setValue(config.speech_pad_ms)
        self.repetition_penalty_spin.setValue(config.repetition_penalty)
        self.no_repeat_ngram_spin.setValue(config.no_repeat_ngram_size)
        self.patience_spin.setValue(config.patience)
        self.length_penalty_spin.setValue(config.length_penalty)
        self.hallucination_silence_spin.setValue(config.hallucination_silence_threshold)

    def _apply_grammar_controls(self, config: GrammarConfig) -> None:
        """Push grammar settings into the UI controls."""
        self.grammar_enhance_checkbox.setChecked(config.enabled)
        self._set_combo_data(self.grammar_backend_combo, config.backend)
        self.grammar_language_input.setText(config.language)
        self.gector_batch_spin.setValue(config.gector_batch_size)
        self.gector_iterations_spin.setValue(config.gector_iterations)

    def _apply_selected_preset_to_controls(self) -> None:
        """Apply the active preset to the visible runtime controls."""
        config = self._build_transcription_config(apply_selected_preset=False)
        preset_key = self.preset_combo.currentData() or DEFAULT_PRESET
        apply_preset(config, str(preset_key))
        self._apply_transcription_controls(config)
        self.update_status(
            f"Preset loaded: {self.preset_combo.currentText()}",
            self.theme.colors.info,
        )

    def _append_runtime_summary(
        self,
        *,
        source: str,
        transcription_config: TranscriptionConfig,
        grammar_config: GrammarConfig,
    ) -> None:
        """Add a concise runtime summary to the progress pane before work starts."""
        self.append_progress(
            f"{source} runtime: backend={transcription_config.batch_backend} | "
            f"openai_model={transcription_config.openai_batch_model} | "
            f"whisper_model={transcription_config.whisper_model} | "
            f"device_pref={transcription_config.device_preference} | "
            f"compute={transcription_config.compute_type} | "
            f"beam={transcription_config.beam_size} | "
            f"batch={transcription_config.batch_size} | "
            f"cpu_fallback_batch={transcription_config.cpu_fallback_batch_size}\n"
        )
        self.append_progress(
            f"{source} safeguards: vad={transcription_config.vad_filter} | "
            f"no_speech={transcription_config.no_speech_threshold:.2f} | "
            f"hallucination_filter={transcription_config.filter_hallucinations} | "
            f"dedupe={transcription_config.deduplicate_repeated_segments} | "
            f"grammar={grammar_config.enabled} ({grammar_config.backend})\n"
        )

    def _load_settings(self) -> None:
        """Load saved settings from previous session."""
        last_url = self.config.ui.last_youtube_url
        if last_url:
            self.url_input.setText(last_url)

        format_index = self.format_combo.findData(self.config.ui.output_format)
        if format_index >= 0:
            self.format_combo.setCurrentIndex(format_index)

        last_mic = self.config.recording.default_microphone
        if last_mic:
            mic_index = self.mic_combo.findText(last_mic)
            if mic_index >= 0:
                self.mic_combo.setCurrentIndex(mic_index)

        last_preset = self.config.ui.transcription_preset or DEFAULT_PRESET
        self.preset_combo.blockSignals(True)
        preset_idx = self.preset_combo.findData(last_preset)
        if preset_idx >= 0:
            self.preset_combo.setCurrentIndex(preset_idx)
        self.preset_combo.blockSignals(False)

        self._apply_transcription_controls(self.config.transcription)
        self._apply_grammar_controls(self.config.grammar)
        self.gpu_memory_spin.setValue(self.config.gpu_memory_fraction)
        self._update_grammar_status()

        gui_logger.debug("Settings loaded from previous session")

    def _save_settings(self) -> None:
        """Save settings for next session."""
        settings = QtCore.QSettings("AnthropicClaude", "YouTubeTranscriber")

        settings.setValue("splitter/state", self.content_splitter.saveState())
        self.config.ui.last_youtube_url = self.url_input.text().strip()
        self.config.ui.output_format = str(self.format_combo.currentData() or "plain")
        self.config.recording.default_microphone = self.mic_combo.currentText()
        self.config.ui.transcription_preset = str(self.preset_combo.currentData() or DEFAULT_PRESET)
        self.config.transcription = self._build_transcription_config()
        self.config.grammar = self._build_grammar_config()
        self.config.gpu_memory_fraction = self.gpu_memory_spin.value()
        save_config()

        gui_logger.debug("Settings saved")

    def _schedule_startup_tasks(self) -> None:
        """Defer non-essential startup work until after the window paints."""
        QtCore.QTimer.singleShot(0, self.populate_microphones)
        QtCore.QTimer.singleShot(0, self._start_startup_checks_thread)

    def _start_startup_checks_thread(self) -> None:
        """Run slower startup diagnostics in the background."""
        threading.Thread(target=self._check_dependencies_on_startup, daemon=True).start()

    def _check_dependencies_on_startup(self) -> None:
        """Check for missing dependencies without blocking the initial UI render."""
        try:
            all_ok, missing = check_dependencies(include_gpu_probe=False)
        except Exception as exc:
            gui_logger.warning("Startup dependency check failed: %s", exc)
            self.output_queue.put(("startup_dependency_check_failed", str(exc)))
        else:
            if not all_ok:
                self.output_queue.put(("startup_missing_dependencies", tuple(missing)))
                gui_logger.warning("Missing dependencies: %s", ", ".join(missing))

        self._detect_gpu_on_startup()

    def _detect_gpu_on_startup(self) -> None:
        """Detect GPU availability and queue a UI-safe status update."""
        try:
            whisper_cuda_ok, gpu_name = get_whisper_cuda_status()
            if whisper_cuda_ok:
                if not gpu_name:
                    gpu_name = "CTranslate2 CUDA"
                self.output_queue.put(("gpu_status", ("cuda", gpu_name)))
                gui_logger.info(f"Whisper CUDA backend detected: {gpu_name}")
            else:
                self.output_queue.put(("gpu_status", ("cpu", "")))
                gui_logger.info("Whisper CUDA backend not detected - using CPU mode")
        except Exception as e:
            self.output_queue.put(("gpu_status_unknown", str(e)))
            gui_logger.warning(f"Failed to detect GPU: {e}")

    def _build_ui(self) -> None:
        """Build the main UI with glassmorphic components and animations."""
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(6)

        # Title section - compact
        title = QtWidgets.QLabel("Speech Transcriber & Recording")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        title.setStyleSheet(self.theme.get_title_style())
        main_layout.addWidget(title)

        # Input cards section
        self._youtube_card = self._build_youtube_card()
        self._local_file_card = self._build_local_file_card()
        self._speech_card = self._build_speech_card()
        self._transcription_settings_card = self._build_transcription_settings_card()

        main_layout.addWidget(self._youtube_card)
        main_layout.addWidget(self._local_file_card)
        main_layout.addWidget(self._speech_card)
        main_layout.addWidget(self._transcription_settings_card)

        # Content splitter with responsive layout
        self.content_splitter = ResponsiveSplitter(QtCore.Qt.Orientation.Vertical, self)
        self.content_splitter.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        # Build content cards
        self._progress_card = self._build_progress_card()
        self._transcript_card = self._build_transcript_card()

        self.content_splitter.addWidget(self._progress_card)
        self.content_splitter.addWidget(self._transcript_card)

        # Restore splitter state or set defaults
        settings = QtCore.QSettings("AnthropicClaude", "YouTubeTranscriber")
        splitter_state = settings.value("splitter/state")
        if splitter_state:
            self.content_splitter.restoreState(splitter_state)
        else:
            self.content_splitter.setDefaultSizes([0.45, 0.55])

        main_layout.addWidget(self.content_splitter, stretch=1)
        main_layout.addWidget(self._build_controls_row())

        self._build_status_bar()

        # Apply main stylesheet
        self.setStyleSheet(self.theme.get_main_stylesheet())

    def _build_status_bar(self) -> None:
        """Build the status bar."""
        status_bar = QtWidgets.QStatusBar(self)
        status_bar.setSizeGripEnabled(False)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.status_label.setStyleSheet(self.theme.get_status_style("success"))

        self.gpu_status_label = QtWidgets.QLabel("GPU: Detecting...")
        self.gpu_status_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.gpu_status_label.setStyleSheet(self.theme.get_gpu_status_style(has_gpu=False))

        self.openai_status_label = QtWidgets.QLabel("")
        self.openai_status_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self._refresh_openai_status()

        self.recording_duration_label = QtWidgets.QLabel("")
        self.recording_duration_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.recording_duration_label.setStyleSheet(
            f"color: {self.theme.colors.recording}; font-weight: 600; font-family: 'Consolas', monospace;"
        )
        self.recording_duration_label.setVisible(False)

        status_bar.addWidget(self.status_label, 1)
        status_bar.addWidget(self.gpu_status_label)
        status_bar.addWidget(self.openai_status_label)
        status_bar.addWidget(self.recording_duration_label)
        self.setStatusBar(status_bar)

    def _refresh_openai_status(self) -> None:
        if is_openai_api_configured():
            self.openai_status_label.setText("OpenAI: Ready")
            self.openai_status_label.setStyleSheet(self.theme.get_gpu_status_style(has_gpu=True))
        else:
            self.openai_status_label.setText("OpenAI: Missing Key")
            self.openai_status_label.setStyleSheet(self.theme.get_gpu_status_style(has_gpu=False))

    def _build_youtube_card(self) -> MaterialCard:
        """Build the YouTube transcription card."""
        card = MaterialCard("YouTube Transcription", self, elevation=1)
        layout = QtWidgets.QGridLayout()
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        # URL input row
        url_label = QtWidgets.QLabel("URL:")
        url_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.url_input = QtWidgets.QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=VIDEO_ID")

        self.transcribe_button = MaterialButton("Transcribe", "primary")
        self.transcribe_button.setToolTip("Transcribe YouTube video (Ctrl+T)")

        self.cancel_youtube_button = MaterialButton("Cancel", "error")
        self.cancel_youtube_button.setToolTip("Cancel YouTube transcription")
        self.cancel_youtube_button.setVisible(False)

        # Format dropdown
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItem("Plain Text", "plain")
        self.format_combo.addItem("Timestamps", "timestamped")
        self.format_combo.setToolTip("Output format")
        self.format_combo.setFixedWidth(110)

        layout.addWidget(url_label, 0, 0)
        layout.addWidget(self.url_input, 0, 1)
        layout.addWidget(self.format_combo, 0, 2)
        layout.addWidget(self.transcribe_button, 0, 3)
        layout.addWidget(self.cancel_youtube_button, 0, 4)

        layout.setColumnStretch(1, 1)
        card.addLayout(layout)
        card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)
        return card

    def _build_speech_card(self) -> MaterialCard:
        """Build the microphone recording card."""
        card = MaterialCard("Microphone Recording", self, elevation=1)
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(10)

        # Microphone selection
        mic_label = QtWidgets.QLabel("Mic:")

        self.mic_combo = QtWidgets.QComboBox()
        self.mic_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.mic_combo.setMinimumContentsLength(25)
        self.mic_combo.addItem("Detecting microphones...")

        self.refresh_mics_button = QtWidgets.QToolButton()
        style = QtWidgets.QApplication.style()
        if style is not None:
            self.refresh_mics_button.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_mics_button.setToolTip("Refresh microphone list")
        self.refresh_mics_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        # Auto-detect best microphone button
        self.auto_mic_button = QtWidgets.QToolButton()
        self.auto_mic_button.setText("Auto")
        self.auto_mic_button.setToolTip("Auto-detect best microphone (filters out virtual devices)")
        self.auto_mic_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        # Record button and status
        self.record_button = MaterialButton("Start Recording", "success")
        self.record_status_label = QtWidgets.QLabel("Live OpenAI realtime transcription")
        self.record_status_label.setStyleSheet(self.theme.get_recording_status_style(is_recording=False))

        self._set_record_button_state(recording=False)

        layout.addWidget(mic_label)
        layout.addWidget(self.mic_combo, stretch=1)
        layout.addWidget(self.auto_mic_button)
        layout.addWidget(self.refresh_mics_button)
        layout.addWidget(self.record_button)
        layout.addWidget(self.record_status_label)

        card.addLayout(layout)
        card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)
        return card

    def _build_transcription_settings_card(self) -> MaterialCard:
        """Build the shared transcription settings card (applies to all transcription sources)."""
        card = MaterialCard("Transcription Settings", self, elevation=1)
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(6)

        def _add_labeled_widget(
            layout: QtWidgets.QGridLayout,
            *,
            row: int,
            column_pair: int,
            label_text: str,
            widget: QtWidgets.QWidget,
            tooltip: str,
        ) -> None:
            widget.setToolTip(tooltip)
            layout.addWidget(self._create_setting_label_with_help(label_text, tooltip), row, column_pair * 2)
            layout.addWidget(widget, row, column_pair * 2 + 1)

        # Row 1: preset + backend/model + language + hotwords
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(10)

        preset_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Preset:"]
        backend_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Backend:"]
        openai_model_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["OpenAI Model:"]
        model_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Model:"]
        language_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Language:"]
        hotwords_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Hotwords:"]
        grammar_fix_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Grammar Fix"]
        filler_cleanup_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Clean Fillers"]
        noise_reduction_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Noise Reduction"]
        normalize_audio_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Normalize Audio"]
        hallucination_filter_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Filter Hallucinations"]
        deduplicate_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Deduplicate Loops"]
        word_timestamps_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Word Timestamps"]
        context_carryover_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Context Carryover"]
        use_vad_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Use VAD"]
        device_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Device:"]
        compute_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Compute:"]
        batch_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Batch:"]
        cpu_fallback_batch_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["CPU Fallback Batch:"]
        beam_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Beam:"]
        temperature_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Temperature:"]
        gpu_memory_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["GPU Memory %:"]
        no_speech_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["No-Speech Threshold:"]
        vad_threshold_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["VAD Threshold:"]
        min_speech_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Min Speech (ms):"]
        min_silence_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Min Silence (ms):"]
        speech_pad_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Speech Pad (ms):"]
        repetition_penalty_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Repetition Penalty:"]
        no_repeat_ngram_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["No-Repeat N-gram:"]
        patience_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Patience:"]
        length_penalty_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Length Penalty:"]
        hallucination_silence_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Hallucination Silence:"]
        grammar_backend_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Grammar Backend:"]
        grammar_language_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["Grammar Language:"]
        gector_batch_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["GECToR Batch:"]
        gector_iterations_tooltip = TRANSCRIPTION_SETTING_HELP_TEXTS["GECToR Iterations:"]

        preset_label = self._create_setting_label_with_help("Preset:", preset_tooltip)
        self.preset_combo = QtWidgets.QComboBox()
        for key, preset in ACCURACY_PRESETS.items():
            self.preset_combo.addItem(preset.name, key)
        default_idx = self.preset_combo.findData(DEFAULT_PRESET)
        if default_idx >= 0:
            self.preset_combo.setCurrentIndex(default_idx)
        self.preset_combo.setToolTip(preset_tooltip)
        self.preset_combo.setFixedWidth(165)

        backend_label = self._create_setting_label_with_help("Backend:", backend_tooltip)
        self.batch_backend_combo = QtWidgets.QComboBox()
        for value, label in BATCH_BACKEND_OPTIONS:
            self.batch_backend_combo.addItem(label, value)
        self.batch_backend_combo.setToolTip(backend_tooltip)
        self.batch_backend_combo.setFixedWidth(135)

        openai_model_label = self._create_setting_label_with_help("OpenAI Model:", openai_model_tooltip)
        self.openai_batch_model_combo = QtWidgets.QComboBox()
        for value, label in OPENAI_BATCH_MODEL_OPTIONS:
            self.openai_batch_model_combo.addItem(label, value)
        self.openai_batch_model_combo.setToolTip(openai_model_tooltip)
        self.openai_batch_model_combo.setFixedWidth(180)

        model_label = self._create_setting_label_with_help("Model:", model_tooltip)
        self.whisper_model_combo = QtWidgets.QComboBox()
        for value, label in WHISPER_MODEL_OPTIONS:
            self.whisper_model_combo.addItem(label, value)
        self.whisper_model_combo.setToolTip(model_tooltip)
        self.whisper_model_combo.setFixedWidth(170)

        language_label = self._create_setting_label_with_help("Language:", language_tooltip)
        self.language_input = QtWidgets.QLineEdit()
        self.language_input.setPlaceholderText("Auto")
        self.language_input.setToolTip(language_tooltip)

        hotwords_label = self._create_setting_label_with_help("Hotwords:", hotwords_tooltip)
        self.hotwords_input = QtWidgets.QLineEdit()
        self.hotwords_input.setPlaceholderText("Optional: meeting terms, names, jargon")
        self.hotwords_input.setToolTip(hotwords_tooltip)

        row1.addWidget(preset_label)
        row1.addWidget(self.preset_combo)
        row1.addWidget(backend_label)
        row1.addWidget(self.batch_backend_combo)
        row1.addWidget(openai_model_label)
        row1.addWidget(self.openai_batch_model_combo)
        row1.addWidget(model_label)
        row1.addWidget(self.whisper_model_combo)
        row1.addWidget(language_label)
        row1.addWidget(self.language_input)
        row1.addWidget(hotwords_label)
        row1.addWidget(self.hotwords_input, stretch=1)

        # Row 2: Output cleanup and grammar toggles
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(10)

        self.grammar_enhance_checkbox = QtWidgets.QCheckBox("Grammar Fix")
        self.grammar_enhance_checkbox.setChecked(self.config.grammar.enabled)
        self.grammar_enhance_checkbox.setToolTip(grammar_fix_tooltip)

        self.grammar_status_label = QtWidgets.QLabel("")
        self.grammar_status_label.setStyleSheet(f"color: {self.theme.colors.text_secondary}; font-size: 11px;")

        self.filler_cleanup_checkbox = QtWidgets.QCheckBox("Clean Fillers")
        self.filler_cleanup_checkbox.setChecked(self.config.transcription.clean_filler_words)
        self.filler_cleanup_checkbox.setToolTip(filler_cleanup_tooltip)

        self.noise_reduction_checkbox = QtWidgets.QCheckBox("Noise Reduction")
        self.noise_reduction_checkbox.setChecked(self.config.transcription.noise_reduction_enabled)
        self.noise_reduction_checkbox.setToolTip(noise_reduction_tooltip)

        self.normalize_audio_checkbox = QtWidgets.QCheckBox("Normalize Audio")
        self.normalize_audio_checkbox.setChecked(self.config.transcription.normalize_audio)
        self.normalize_audio_checkbox.setToolTip(normalize_audio_tooltip)

        self.filter_hallucinations_checkbox = QtWidgets.QCheckBox("Filter Hallucinations")
        self.filter_hallucinations_checkbox.setChecked(self.config.transcription.filter_hallucinations)
        self.filter_hallucinations_checkbox.setToolTip(hallucination_filter_tooltip)

        self.deduplicate_checkbox = QtWidgets.QCheckBox("Deduplicate Loops")
        self.deduplicate_checkbox.setChecked(self.config.transcription.deduplicate_repeated_segments)
        self.deduplicate_checkbox.setToolTip(deduplicate_tooltip)

        row2.addWidget(self._wrap_setting_widget_with_help(self.grammar_enhance_checkbox, grammar_fix_tooltip))
        row2.addWidget(self.grammar_status_label)
        row2.addWidget(self._wrap_setting_widget_with_help(self.filler_cleanup_checkbox, filler_cleanup_tooltip))
        row2.addWidget(self._wrap_setting_widget_with_help(self.noise_reduction_checkbox, noise_reduction_tooltip))
        row2.addWidget(self._wrap_setting_widget_with_help(self.normalize_audio_checkbox, normalize_audio_tooltip))
        row2.addWidget(
            self._wrap_setting_widget_with_help(self.filter_hallucinations_checkbox, hallucination_filter_tooltip)
        )
        row2.addWidget(self._wrap_setting_widget_with_help(self.deduplicate_checkbox, deduplicate_tooltip))
        row2.addStretch(1)

        row3 = QtWidgets.QHBoxLayout()
        row3.setSpacing(10)

        self.word_timestamps_checkbox = QtWidgets.QCheckBox("Word Timestamps")
        self.word_timestamps_checkbox.setChecked(self.config.transcription.word_timestamps)
        self.word_timestamps_checkbox.setToolTip(word_timestamps_tooltip)

        self.previous_text_checkbox = QtWidgets.QCheckBox("Context Carryover")
        self.previous_text_checkbox.setChecked(self.config.transcription.condition_on_previous_text)
        self.previous_text_checkbox.setToolTip(context_carryover_tooltip)

        self.vad_filter_checkbox = QtWidgets.QCheckBox("Use VAD")
        self.vad_filter_checkbox.setChecked(self.config.transcription.vad_filter)
        self.vad_filter_checkbox.setToolTip(use_vad_tooltip)

        row3.addWidget(self._wrap_setting_widget_with_help(self.word_timestamps_checkbox, word_timestamps_tooltip))
        row3.addWidget(self._wrap_setting_widget_with_help(self.previous_text_checkbox, context_carryover_tooltip))
        row3.addWidget(self._wrap_setting_widget_with_help(self.vad_filter_checkbox, use_vad_tooltip))
        row3.addStretch(1)

        advanced_label = QtWidgets.QLabel("Advanced Runtime")
        advanced_label.setStyleSheet(
            f"color: {self.theme.colors.text_secondary}; font-weight: 600; letter-spacing: 0.5px;"
        )

        advanced_grid = QtWidgets.QGridLayout()
        advanced_grid.setHorizontalSpacing(10)
        advanced_grid.setVerticalSpacing(6)

        self.device_preference_combo = QtWidgets.QComboBox()
        for value, label in DEVICE_PREFERENCE_OPTIONS:
            self.device_preference_combo.addItem(label, value)
        self.device_preference_combo.setToolTip(device_tooltip)

        self.compute_type_combo = QtWidgets.QComboBox()
        for value, label in COMPUTE_TYPE_OPTIONS:
            self.compute_type_combo.addItem(label, value)
        self.compute_type_combo.setToolTip(compute_tooltip)

        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setToolTip(batch_tooltip)

        self.cpu_fallback_batch_spin = QtWidgets.QSpinBox()
        self.cpu_fallback_batch_spin.setRange(1, 64)
        self.cpu_fallback_batch_spin.setToolTip(cpu_fallback_batch_tooltip)

        self.beam_size_spin = QtWidgets.QSpinBox()
        self.beam_size_spin.setRange(1, 10)
        self.beam_size_spin.setToolTip(beam_tooltip)

        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 1.0)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setToolTip(temperature_tooltip)

        self.gpu_memory_spin = QtWidgets.QDoubleSpinBox()
        self.gpu_memory_spin.setRange(0.05, 1.0)
        self.gpu_memory_spin.setDecimals(2)
        self.gpu_memory_spin.setSingleStep(0.05)
        self.gpu_memory_spin.setToolTip(gpu_memory_tooltip)

        self.no_speech_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.no_speech_threshold_spin.setRange(0.0, 1.0)
        self.no_speech_threshold_spin.setDecimals(2)
        self.no_speech_threshold_spin.setSingleStep(0.05)
        self.no_speech_threshold_spin.setToolTip(no_speech_tooltip)

        self.vad_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.vad_threshold_spin.setRange(0.0, 1.0)
        self.vad_threshold_spin.setDecimals(2)
        self.vad_threshold_spin.setSingleStep(0.05)
        self.vad_threshold_spin.setToolTip(vad_threshold_tooltip)

        self.min_speech_duration_spin = QtWidgets.QSpinBox()
        self.min_speech_duration_spin.setRange(10, 10000)
        self.min_speech_duration_spin.setSingleStep(10)
        self.min_speech_duration_spin.setToolTip(min_speech_tooltip)

        self.min_silence_duration_spin = QtWidgets.QSpinBox()
        self.min_silence_duration_spin.setRange(10, 20000)
        self.min_silence_duration_spin.setSingleStep(50)
        self.min_silence_duration_spin.setToolTip(min_silence_tooltip)

        self.speech_pad_spin = QtWidgets.QSpinBox()
        self.speech_pad_spin.setRange(0, 5000)
        self.speech_pad_spin.setSingleStep(25)
        self.speech_pad_spin.setToolTip(speech_pad_tooltip)

        self.repetition_penalty_spin = QtWidgets.QDoubleSpinBox()
        self.repetition_penalty_spin.setRange(1.0, 5.0)
        self.repetition_penalty_spin.setDecimals(2)
        self.repetition_penalty_spin.setSingleStep(0.1)
        self.repetition_penalty_spin.setToolTip(repetition_penalty_tooltip)

        self.no_repeat_ngram_spin = QtWidgets.QSpinBox()
        self.no_repeat_ngram_spin.setRange(0, 10)
        self.no_repeat_ngram_spin.setToolTip(no_repeat_ngram_tooltip)

        self.patience_spin = QtWidgets.QDoubleSpinBox()
        self.patience_spin.setRange(0.0, 5.0)
        self.patience_spin.setDecimals(2)
        self.patience_spin.setSingleStep(0.1)
        self.patience_spin.setToolTip(patience_tooltip)

        self.length_penalty_spin = QtWidgets.QDoubleSpinBox()
        self.length_penalty_spin.setRange(0.0, 5.0)
        self.length_penalty_spin.setDecimals(2)
        self.length_penalty_spin.setSingleStep(0.1)
        self.length_penalty_spin.setToolTip(length_penalty_tooltip)

        self.hallucination_silence_spin = QtWidgets.QDoubleSpinBox()
        self.hallucination_silence_spin.setRange(0.0, 5.0)
        self.hallucination_silence_spin.setDecimals(2)
        self.hallucination_silence_spin.setSingleStep(0.1)
        self.hallucination_silence_spin.setToolTip(hallucination_silence_tooltip)

        self.grammar_backend_combo = QtWidgets.QComboBox()
        for value, label in GRAMMAR_BACKEND_OPTIONS:
            self.grammar_backend_combo.addItem(label, value)
        self.grammar_backend_combo.setToolTip(grammar_backend_tooltip)

        self.grammar_language_input = QtWidgets.QLineEdit()
        self.grammar_language_input.setPlaceholderText("en-US")
        self.grammar_language_input.setToolTip(grammar_language_tooltip)

        self.gector_batch_spin = QtWidgets.QSpinBox()
        self.gector_batch_spin.setRange(1, 64)
        self.gector_batch_spin.setToolTip(gector_batch_tooltip)

        self.gector_iterations_spin = QtWidgets.QSpinBox()
        self.gector_iterations_spin.setRange(1, 10)
        self.gector_iterations_spin.setToolTip(gector_iterations_tooltip)

        _add_labeled_widget(
            advanced_grid, row=0, column_pair=0, label_text="Device:", widget=self.device_preference_combo, tooltip=device_tooltip
        )
        _add_labeled_widget(
            advanced_grid, row=0, column_pair=1, label_text="Compute:", widget=self.compute_type_combo, tooltip=compute_tooltip
        )
        _add_labeled_widget(
            advanced_grid, row=0, column_pair=2, label_text="Batch:", widget=self.batch_size_spin, tooltip=batch_tooltip
        )
        _add_labeled_widget(
            advanced_grid,
            row=0,
            column_pair=3,
            label_text="CPU Fallback Batch:",
            widget=self.cpu_fallback_batch_spin,
            tooltip=cpu_fallback_batch_tooltip,
        )

        _add_labeled_widget(
            advanced_grid, row=1, column_pair=0, label_text="Beam:", widget=self.beam_size_spin, tooltip=beam_tooltip
        )
        _add_labeled_widget(
            advanced_grid,
            row=1,
            column_pair=1,
            label_text="Temperature:",
            widget=self.temperature_spin,
            tooltip=temperature_tooltip,
        )
        _add_labeled_widget(
            advanced_grid, row=1, column_pair=2, label_text="GPU Memory %:", widget=self.gpu_memory_spin, tooltip=gpu_memory_tooltip
        )
        _add_labeled_widget(
            advanced_grid,
            row=1,
            column_pair=3,
            label_text="No-Speech Threshold:",
            widget=self.no_speech_threshold_spin,
            tooltip=no_speech_tooltip,
        )

        _add_labeled_widget(
            advanced_grid,
            row=2,
            column_pair=0,
            label_text="VAD Threshold:",
            widget=self.vad_threshold_spin,
            tooltip=vad_threshold_tooltip,
        )
        _add_labeled_widget(
            advanced_grid,
            row=2,
            column_pair=1,
            label_text="Min Speech (ms):",
            widget=self.min_speech_duration_spin,
            tooltip=min_speech_tooltip,
        )
        _add_labeled_widget(
            advanced_grid,
            row=2,
            column_pair=2,
            label_text="Min Silence (ms):",
            widget=self.min_silence_duration_spin,
            tooltip=min_silence_tooltip,
        )
        _add_labeled_widget(
            advanced_grid,
            row=2,
            column_pair=3,
            label_text="Speech Pad (ms):",
            widget=self.speech_pad_spin,
            tooltip=speech_pad_tooltip,
        )

        _add_labeled_widget(
            advanced_grid,
            row=3,
            column_pair=0,
            label_text="Repetition Penalty:",
            widget=self.repetition_penalty_spin,
            tooltip=repetition_penalty_tooltip,
        )
        _add_labeled_widget(
            advanced_grid,
            row=3,
            column_pair=1,
            label_text="No-Repeat N-gram:",
            widget=self.no_repeat_ngram_spin,
            tooltip=no_repeat_ngram_tooltip,
        )
        _add_labeled_widget(
            advanced_grid, row=3, column_pair=2, label_text="Patience:", widget=self.patience_spin, tooltip=patience_tooltip
        )
        _add_labeled_widget(
            advanced_grid,
            row=3,
            column_pair=3,
            label_text="Length Penalty:",
            widget=self.length_penalty_spin,
            tooltip=length_penalty_tooltip,
        )

        _add_labeled_widget(
            advanced_grid,
            row=4,
            column_pair=0,
            label_text="Hallucination Silence:",
            widget=self.hallucination_silence_spin,
            tooltip=hallucination_silence_tooltip,
        )
        _add_labeled_widget(
            advanced_grid,
            row=4,
            column_pair=1,
            label_text="Grammar Backend:",
            widget=self.grammar_backend_combo,
            tooltip=grammar_backend_tooltip,
        )
        _add_labeled_widget(
            advanced_grid,
            row=4,
            column_pair=2,
            label_text="Grammar Language:",
            widget=self.grammar_language_input,
            tooltip=grammar_language_tooltip,
        )
        _add_labeled_widget(
            advanced_grid,
            row=4,
            column_pair=3,
            label_text="GECToR Batch:",
            widget=self.gector_batch_spin,
            tooltip=gector_batch_tooltip,
        )

        _add_labeled_widget(
            advanced_grid,
            row=5,
            column_pair=0,
            label_text="GECToR Iterations:",
            widget=self.gector_iterations_spin,
            tooltip=gector_iterations_tooltip,
        )

        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addLayout(row3)
        main_layout.addWidget(advanced_label)
        main_layout.addLayout(advanced_grid)

        card.addLayout(main_layout)
        card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)
        self._update_grammar_status()
        return card

    def _update_grammar_status(self) -> None:
        """Update the grammar status indicator."""
        if not self.grammar_enhance_checkbox.isChecked():
            self.grammar_status_label.setText("(Off)")
            self.grammar_status_label.setStyleSheet(
                f"color: {self.theme.colors.text_secondary}; font-size: 11px;"
            )
            return

        grammar_config = self._build_grammar_config()
        is_available, status = check_grammar_status(lazy=True, config=grammar_config)
        if is_available:
            self.grammar_status_label.setText(f"({status})")
            if "GECToR" in status:
                self.grammar_status_label.setStyleSheet(f"color: {self.theme.colors.success}; font-size: 11px;")
            else:
                self.grammar_status_label.setStyleSheet(f"color: {self.theme.colors.info}; font-size: 11px;")
        else:
            self.grammar_status_label.setText(f"({status})")
            self.grammar_status_label.setStyleSheet(f"color: {self.theme.colors.warning}; font-size: 11px;")

    def _build_local_file_card(self) -> MaterialCard:
        """Build the local file transcription card."""
        card = MaterialCard("Local File", self, elevation=1)
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(10)

        # File selection
        file_label = QtWidgets.QLabel("File:")

        self.file_path_input = QtWidgets.QLineEdit()
        self.file_path_input.setPlaceholderText("Select audio/video file...")
        self.file_path_input.setReadOnly(True)

        self.browse_file_button = MaterialButton("Browse", "secondary")
        self.browse_file_button.setToolTip("Select audio/video file")

        self.transcribe_file_button = MaterialButton("Transcribe", "primary")
        self.transcribe_file_button.setToolTip("Transcribe selected file")
        self.transcribe_file_button.setEnabled(False)

        layout.addWidget(file_label)
        layout.addWidget(self.file_path_input, stretch=1)
        layout.addWidget(self.browse_file_button)
        layout.addWidget(self.transcribe_file_button)

        card.addLayout(layout)
        card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)
        return card

    def _build_progress_card(self) -> MaterialCard:
        """Build the progress output card."""
        card = MaterialCard("Progress Log", self, elevation=1)

        self.progress_output = QtWidgets.QPlainTextEdit()
        self.progress_output.setObjectName("ProgressOutput")
        self.progress_output.setReadOnly(True)
        self.progress_output.setPlaceholderText("Verbose runtime output appears here...")
        self.progress_output.setMinimumHeight(120)
        self.progress_output.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        card.addWidget(self.progress_output, stretch=1)
        return card

    def _build_transcript_card(self) -> MaterialCard:
        """Build the final transcript card."""
        card = MaterialCard("Transcript", self, elevation=1)

        self.transcript_edit = QtWidgets.QTextEdit()
        self.transcript_edit.setObjectName("TranscriptOutput")
        self.transcript_edit.setPlaceholderText("Transcript appears here...")
        self.transcript_edit.setMinimumHeight(80)
        self.transcript_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        card.addWidget(self.transcript_edit, stretch=1)
        return card

    def _build_controls_row(self) -> QtWidgets.QWidget:
        """Build the bottom controls row."""
        container = QtWidgets.QWidget(self)
        container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.clear_button = MaterialButton("Clear", "warning")
        self.clear_button.setToolTip("Clear all text outputs (Ctrl+L)")

        self.save_button = MaterialButton("Save to File", "success")
        self.save_button.setToolTip("Save transcript to file (Ctrl+S)")

        self.copy_button = MaterialButton("Copy to Clipboard", "primary")
        self.copy_button.setToolTip("Copy transcript to clipboard (Ctrl+Shift+C)")

        self.exit_button = MaterialButton("Exit", "error")
        self.exit_button.setToolTip("Close application")

        layout.addWidget(self.clear_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.copy_button)
        layout.addStretch(1)
        layout.addWidget(self.exit_button)

        return container

    def _set_record_button_state(self, *, recording: bool) -> None:
        """Toggle record button visuals."""
        if recording:
            self.record_button.setText("Stop Recording")
            self.record_button.setVariant("error")
            self.record_button.setToolTip("Stop recording (Ctrl+R)")
            self.record_status_label.setText("RECORDING - Speak now!")
            self.record_status_label.setStyleSheet(self.theme.get_recording_status_style(is_recording=True))
        else:
            self.record_button.setText("Start Recording")
            self.record_button.setVariant("success")
            self.record_button.setToolTip("Start live OpenAI speech-to-text (Ctrl+R)")
            self.record_status_label.setText("Live OpenAI realtime transcription")
            self.record_status_label.setStyleSheet(self.theme.get_recording_status_style(is_recording=False))

    def _connect_signals(self) -> None:
        """Connect UI signals to handlers."""
        self.transcribe_button.clicked.connect(self.start_youtube_transcription)
        self.cancel_youtube_button.clicked.connect(self.cancel_youtube_transcription)
        self.browse_file_button.clicked.connect(self.browse_local_file)
        self.transcribe_file_button.clicked.connect(self.start_local_file_transcription)
        self.record_button.clicked.connect(self.toggle_recording)
        self.refresh_mics_button.clicked.connect(self.populate_microphones)
        self.auto_mic_button.clicked.connect(self.auto_detect_microphone)
        self.clear_button.clicked.connect(self.handle_clear)
        self.save_button.clicked.connect(self.handle_save)
        self.copy_button.clicked.connect(self.handle_copy)
        self.exit_button.clicked.connect(self.close)
        self.preset_combo.currentIndexChanged.connect(self._apply_selected_preset_to_controls)
        self.grammar_enhance_checkbox.stateChanged.connect(self._update_grammar_status)
        self.grammar_backend_combo.currentIndexChanged.connect(self._update_grammar_status)
        self.gpu_memory_spin.valueChanged.connect(self._sync_runtime_preferences_from_controls)

        # Keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self).activated.connect(self.toggle_recording)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self).activated.connect(self.handle_save)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+C"), self).activated.connect(self.handle_copy)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self).activated.connect(self.handle_clear)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+T"), self).activated.connect(self.start_youtube_transcription)

    def populate_microphones(self) -> None:
        """Scan for available microphones and update the combo box."""
        current_selection = self.mic_combo.currentText()
        if current_selection == "Detecting microphones...":
            current_selection = ""
        preferred_selection = current_selection or self.config.recording.default_microphone
        microphones = self.get_microphone_list()

        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        self.mic_combo.addItems(microphones)
        self.mic_combo.blockSignals(False)

        if preferred_selection and preferred_selection in microphones:
            index = microphones.index(preferred_selection)
            self.mic_combo.setCurrentIndex(index)
        elif microphones:
            self.mic_combo.setCurrentIndex(0)

        self.update_status(f"Detected {len(microphones)} microphone(s)")

    def get_microphone_list(self) -> list[str]:
        """Return list of available microphone descriptions."""
        try:
            sd = _get_sounddevice_module()
            devices = sd.query_devices()
        except Exception as exc:
            self.output_queue.put(("error", f"Failed to query audio devices: {exc}"))
            return ["Default Microphone"]

        mic_list: list[str] = []
        for idx, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                name = device.get("name", f"Device {idx}")
                mic_list.append(f"{idx}: {name}")

        return mic_list or ["Default Microphone"]

    def auto_detect_microphone(self) -> None:
        """Auto-detect and select the best real microphone, filtering out virtual devices."""
        try:
            sd = _get_sounddevice_module()
            devices = sd.query_devices()
        except Exception as exc:
            self.update_status(f"Failed to query devices: {exc}", self.theme.colors.error)
            return

        # Keywords for scoring microphones (higher score = better)
        # Dedicated USB/XLR microphones (highest priority)
        premium_mics = {
            "pd200x", "podcast", "maono", "blue yeti", "blue snowball",
            "shure", "rode", "audio-technica", "at2020", "samson", "fifine",
            "hyperx", "elgato", "razer seiren", "jlab", "tonor", "condenser",
        }
        # Standard built-in microphones (medium priority)
        standard_mics = {"realtek", "high definition audio"}
        # Webcam microphones (lower priority but still real)
        webcam_mics = {"webcam", "camera", "nexigo", "logitech c"}
        # Virtual/loopback devices to avoid (negative score)
        virtual_keywords = {
            "virtual", "loopback", "sound mapper", "stereo mix", "wave",
            "voicemod", "vb-audio", "cable", "oculus", "vad wave",
            "what u hear", "mix", "desktop audio",
        }
        # Generic system devices (low priority)
        generic_keywords = {"primary sound capture", "default", "input ()"}

        best_mic_idx = -1
        best_score = -999
        best_name = ""

        for idx, device in enumerate(devices):
            if device.get("max_input_channels", 0) < 1:
                continue  # Skip output-only devices

            name = device.get("name", "").lower()
            score = 0

            # Check for virtual devices (heavy penalty)
            if any(vk in name for vk in virtual_keywords):
                score -= 100
            # Check for generic system devices
            elif any(gk in name for gk in generic_keywords):
                score -= 50
            # Check for premium dedicated microphones
            elif any(pm in name for pm in premium_mics):
                score += 100
            # Check for standard built-in mics
            elif any(sm in name for sm in standard_mics):
                score += 30
            # Check for webcam mics
            elif any(wm in name for wm in webcam_mics):
                score += 20
            else:
                # Unknown device - neutral score
                score += 10

            # Bonus for higher sample rates (indicates quality hardware)
            sample_rate = device.get("default_samplerate", 0)
            if sample_rate >= 48000:
                score += 5
            elif sample_rate >= 44100:
                score += 3

            # Slight bonus for being marked as default input
            try:
                default_input = sd.query_devices(kind='input')
                if default_input and default_input.get('name') == device.get('name'):
                    score += 15
            except Exception as exc:
                gui_logger.debug("Could not query default input device for scoring", exc_info=exc)

            if score > best_score:
                best_score = score
                best_mic_idx = idx
                best_name = device.get("name", f"Device {idx}")

        if best_mic_idx >= 0:
            # Find this mic in the combo box
            for i in range(self.mic_combo.count()):
                if self.mic_combo.itemText(i).startswith(f"{best_mic_idx}:"):
                    self.mic_combo.setCurrentIndex(i)
                    self.update_status(f"Auto-selected: {best_name}", self.theme.colors.success_light)
                    return

        self.update_status("No suitable microphone found", self.theme.colors.warning)

    def update_status(self, message: str, color: str = "#66BB6A") -> None:
        """Update the status bar."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _update_recording_duration(self) -> None:
        """Update the recording duration label."""
        if self._recording_start_time is not None:
            elapsed = int(time.time() - self._recording_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.recording_duration_label.setText(f"Recording: {minutes:02d}:{seconds:02d}")
            self._recording_duration_seconds = elapsed

    def _reset_recording_ui_state(self) -> None:
        """Reset timers and transient recording state after capture stops or fails."""
        self._recording_stop_event.set()
        self._recording_start_time = None
        self._recording_duration_seconds = 0
        self._recording_sample_rate = self.config.recording.sample_rate
        self.recording_timer.stop()
        self.recording_duration_label.clear()
        self.recording_duration_label.setVisible(False)
        with self._audio_buffer_lock:
            self._accumulated_audio_buffer.clear()

    def append_progress(self, message: str) -> None:
        """Append a progress message with memory limit."""
        if not message:
            return

        self.progress_output.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.progress_output.insertPlainText(message)

        MAX_LOG_LINES = 15000
        doc = self.progress_output.document()
        if doc and doc.lineCount() > MAX_LOG_LINES:
            cursor = QtGui.QTextCursor(doc)
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
            cursor.movePosition(
                QtGui.QTextCursor.MoveOperation.Down,
                QtGui.QTextCursor.MoveMode.KeepAnchor,
                doc.lineCount() - MAX_LOG_LINES
            )
            cursor.removeSelectedText()

        scrollbar = self.progress_output.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def update_transcript(self, text: str) -> None:
        """Replace final transcript text."""
        self.transcript_edit.setPlainText(text)

    def start_youtube_transcription(self) -> None:
        """Spawn worker thread for YouTube transcription."""
        url = self.url_input.text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(self, "Missing URL", "Please enter a YouTube URL to transcribe.")
            return

        if self.transcribe_thread and self.transcribe_thread.is_alive():
            QtWidgets.QMessageBox.information(self, "Transcription In Progress", "A transcription is already running.")
            return

        output_format = self.format_combo.currentData()

        self.progress_output.clear()
        self.update_status("Validating URL...", self.theme.colors.warning)
        self.transcribe_button.setEnabled(False)
        self.cancel_youtube_button.setVisible(True)
        self._youtube_cancel_event.clear()
        transcription_config = self._build_transcription_config()
        grammar_config = self._build_grammar_config()
        self._append_runtime_summary(
            source="YouTube",
            transcription_config=transcription_config,
            grammar_config=grammar_config,
        )

        thread = threading.Thread(
            target=self.transcribe_youtube_thread,
            args=(url, output_format, transcription_config, grammar_config),
            daemon=True,
        )
        self.transcribe_thread = thread
        try:
            thread.start()
        except RuntimeError as e:
            self.output_queue.put(("error", f"Failed to start transcription thread: {e}"))
            self.output_queue.put(("status", "Ready"))
            self.transcribe_button.setEnabled(True)
            self.cancel_youtube_button.setVisible(False)
            return

    def cancel_youtube_transcription(self) -> None:
        """Cancel the running YouTube transcription."""
        self._youtube_cancel_event.set()
        self.update_status("Cancelling transcription...", self.theme.colors.warning)
        self.cancel_youtube_button.setEnabled(False)
        gui_logger.info("YouTube transcription cancelled by user")

    def _build_transcription_config(self, *, apply_selected_preset: bool = True) -> TranscriptionConfig:
        """Build TranscriptionConfig from current GUI settings."""
        config = TranscriptionConfig(**asdict(self.config.transcription))
        if hasattr(self, "gpu_memory_spin"):
            self.config.gpu_memory_fraction = self.gpu_memory_spin.value()
        if apply_selected_preset:
            preset_key = self.preset_combo.currentData() or DEFAULT_PRESET
            apply_preset(config, str(preset_key))

        config.batch_backend = str(self.batch_backend_combo.currentData() or config.batch_backend)
        config.openai_batch_model = str(self.openai_batch_model_combo.currentData() or config.openai_batch_model)
        config.whisper_model = str(self.whisper_model_combo.currentData() or config.whisper_model)
        language = self.language_input.text().strip()
        config.language = language or None
        hotwords = self.hotwords_input.text().strip()
        config.hotwords = hotwords or None
        config.device_preference = str(self.device_preference_combo.currentData() or config.device_preference)
        config.compute_type = str(self.compute_type_combo.currentData() or config.compute_type)
        config.batch_size = self.batch_size_spin.value()
        config.cpu_fallback_batch_size = self.cpu_fallback_batch_spin.value()
        config.beam_size = self.beam_size_spin.value()
        config.temperature = self.temperature_spin.value()
        config.vad_filter = self.vad_filter_checkbox.isChecked()
        config.no_speech_threshold = self.no_speech_threshold_spin.value()
        config.word_timestamps = self.word_timestamps_checkbox.isChecked()
        config.patience = self.patience_spin.value()
        config.length_penalty = self.length_penalty_spin.value()
        config.hallucination_silence_threshold = self.hallucination_silence_spin.value()
        config.vad_threshold = self.vad_threshold_spin.value()
        config.min_speech_duration_ms = self.min_speech_duration_spin.value()
        config.min_silence_duration_ms = self.min_silence_duration_spin.value()
        config.speech_pad_ms = self.speech_pad_spin.value()
        config.condition_on_previous_text = self.previous_text_checkbox.isChecked()
        config.clean_filler_words = self.filler_cleanup_checkbox.isChecked()
        config.filter_hallucinations = self.filter_hallucinations_checkbox.isChecked()
        config.deduplicate_repeated_segments = self.deduplicate_checkbox.isChecked()
        config.repetition_penalty = self.repetition_penalty_spin.value()
        config.no_repeat_ngram_size = self.no_repeat_ngram_spin.value()
        config.noise_reduction_enabled = self.noise_reduction_checkbox.isChecked()
        config.normalize_audio = self.normalize_audio_checkbox.isChecked()
        return config

    def _build_grammar_config(self) -> GrammarConfig:
        """Build GrammarConfig from current GUI settings."""
        config = GrammarConfig(**asdict(self.config.grammar))
        config.enabled = self.grammar_enhance_checkbox.isChecked()
        config.backend = str(self.grammar_backend_combo.currentData() or config.backend)
        config.language = self.grammar_language_input.text().strip() or "en-US"
        config.gector_batch_size = self.gector_batch_spin.value()
        config.gector_iterations = self.gector_iterations_spin.value()
        return config

    def transcribe_youtube_thread(
        self,
        url: str,
        output_format: str = "plain",
        transcription_config: TranscriptionConfig | None = None,
        grammar_config: GrammarConfig | None = None,
    ) -> None:
        """Background YouTube transcription workflow."""
        transcription_config = transcription_config or self._build_transcription_config()
        grammar_config = grammar_config or self._build_grammar_config()
        transcription_succeeded = False

        try:
            with _worker_queue_bridge(self.output_queue):
                self.output_queue.put(("status", "Validating URL...", self.theme.colors.warning))
                self.output_queue.put(("progress", "Checking the YouTube URL and trying built-in captions first...\n"))

                if self._youtube_cancel_event.is_set():
                    self.output_queue.put(("cancelled", ""))
                    return

                is_valid, error_msg = validate_youtube_url(url)
                if not is_valid:
                    self.output_queue.put(("error", f"Invalid URL: {error_msg}"))
                    return

                video_id = extract_video_id(url)
                if not video_id:
                    self.output_queue.put(("error", "Could not extract video ID from the provided URL."))
                    return

                if self._youtube_cancel_event.is_set():
                    self.output_queue.put(("cancelled", ""))
                    return

                self.output_queue.put(("status", f"Processing video: {video_id}", self.theme.colors.warning))

                gui_logger.info(f"Processing video ID: {video_id}")
                gui_logger.info("-" * 50)

                transcript, segments_data = get_youtube_transcript(video_id)

                if self._youtube_cancel_event.is_set():
                    self.output_queue.put(("cancelled", ""))
                    return

                if not transcript:
                    gui_logger.info("\nNo captions available. Downloading audio and transcribing...")
                    gui_logger.info("-" * 50)
                    backend = transcription_config.batch_backend
                    self.output_queue.put(
                        (
                            "progress",
                            "No captions found. Falling back to audio download + "
                            f"{backend} transcription...\n",
                        )
                    )

                    if self._youtube_cancel_event.is_set():
                        self.output_queue.put(("cancelled", ""))
                        return

                    ffmpeg_location = find_ffmpeg()
                    audio_file = download_audio(url, f"temp_audio_{video_id}", ffmpeg_location)

                    if self._youtube_cancel_event.is_set():
                        if audio_file and os.path.exists(audio_file):
                            os.remove(audio_file)
                        self.output_queue.put(("cancelled", ""))
                        return

                    if not audio_file:
                        self.output_queue.put(("error", "Failed to download audio for transcription."))
                        return

                    if backend in {"openai", "compare"} and not is_openai_api_configured():
                        if audio_file and os.path.exists(audio_file):
                            os.remove(audio_file)
                        self.output_queue.put(("error", "OPENAI_API_KEY is not set. Rotate the exposed key and set a fresh key."))
                        return

                    if backend == "openai":
                        self.output_queue.put(("status", "Transcribing audio with OpenAI...", self.theme.colors.warning))
                        self.output_queue.put(("progress", "Preparing audio download for OpenAI transcription...\n"))
                        transcript, segments_data = transcribe_audio_openai(
                            audio_file,
                            ffmpeg_location,
                            config=transcription_config,
                            cleanup_audio_file=True,
                        )
                    elif backend == "compare":
                        self.output_queue.put(("status", "Comparing OpenAI and local Whisper...", self.theme.colors.warning))
                        self.output_queue.put(("progress", "Running OpenAI transcription first...\n"))
                        openai_transcript, _openai_segments = transcribe_audio_openai(
                            audio_file,
                            ffmpeg_location,
                            config=transcription_config,
                            cleanup_audio_file=False,
                        )

                        self.output_queue.put(("progress", "Running local Whisper transcription for comparison...\n"))
                        unload_gector()
                        torch_module = get_torch(context="gui_transcriber:youtube_empty_cache")
                        if torch_module is not None and torch_module.cuda.is_available():
                            torch_module.cuda.empty_cache()
                        local_transcript, _local_segments = transcribe_audio(
                            audio_file,
                            ffmpeg_location,
                            config=transcription_config,
                        )
                        transcript = _format_backend_comparison(
                            openai_transcript=openai_transcript,
                            local_transcript=local_transcript,
                        )
                        segments_data = None
                    else:
                        self.output_queue.put(("status", "Loading Whisper and transcribing audio...", self.theme.colors.warning))
                        self.output_queue.put(
                            (
                                "progress",
                                "Preparing audio download, freeing grammar-model VRAM, and starting Whisper...\n",
                            )
                        )

                        # Unload GECToR model to free GPU memory for Whisper
                        unload_gector()
                        torch_module = get_torch(context="gui_transcriber:youtube_empty_cache")
                        if torch_module is not None and torch_module.cuda.is_available():
                            torch_module.cuda.empty_cache()

                        transcript, segments_data = transcribe_audio(audio_file, ffmpeg_location, config=transcription_config)

                if self._youtube_cancel_event.is_set():
                    self.output_queue.put(("cancelled", ""))
                    return

                if transcript:
                    # Grammar Post-Processing (if enabled)
                    grammar_enhanced = False
                    if grammar_config.enabled and transcription_config.batch_backend == "compare":
                        self.output_queue.put(("progress", "Grammar correction skipped in Compare mode to preserve raw backend outputs.\n"))
                    elif grammar_config.enabled:
                        grammar_result = _apply_optional_grammar_corrections(
                            self.output_queue,
                            transcript,
                            segments_data if segments_data else [],
                            grammar_config,
                            warning_color=self.theme.colors.warning,
                            start_status="Applying grammar corrections...",
                            start_progress="\nStarting grammar correction...\n",
                            success_progress="Grammar correction applied successfully.\n",
                        )
                        if grammar_result.completed:
                            transcript = grammar_result.transcript
                            segments_data = grammar_result.segments_data
                            grammar_enhanced = grammar_result.grammar_enhanced

                    _queue_transcript_snapshot(
                        self.output_queue,
                        transcript,
                        segments_data,
                        output_format=str(output_format or "plain"),
                    )
                    transcription_succeeded = True

                    status_msg = _build_transcription_complete_status(
                        grammar_enhanced=grammar_enhanced,
                    )
                    self.output_queue.put(("status", status_msg, self.theme.colors.success_light))
                else:
                    self.output_queue.put(("error", "Failed to obtain transcript."))

        except Exception as exc:
            gui_logger.error(f"YouTube transcription error: {exc}", exc_info=True)
            self.output_queue.put(("error", f"Error: {exc}"))
        finally:
            self.output_queue.put(("transcribe_finished", transcription_succeeded))
            self.output_queue.put(("transcribe_thread_done", ""))

    def browse_local_file(self) -> None:
        """Open file dialog to select an audio or video file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Audio/Video File",
            "",
            "All Media Files (*.mp3 *.mp4 *.wav *.m4a *.flac *.ogg *.avi *.mkv *.mov *.webm *.aac *.wma *.opus);;"
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.wma *.opus);;"
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;"
            "All Files (*.*)"
        )

        if file_path:
            self.file_path_input.setText(file_path)
            self.transcribe_file_button.setEnabled(True)
            self.update_status(f"Selected file: {os.path.basename(file_path)}", self.theme.colors.success_light)
            gui_logger.info(f"Selected local file: {file_path}")

    def start_local_file_transcription(self) -> None:
        """Start transcription of the selected local file."""
        file_path = self.file_path_input.text().strip()

        if not file_path:
            QtWidgets.QMessageBox.warning(self, "No File Selected", "Please select a file to transcribe.")
            return

        if not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(self, "File Not Found", f"The selected file does not exist:\n{file_path}")
            return

        if self.transcribe_thread and self.transcribe_thread.is_alive():
            QtWidgets.QMessageBox.information(self, "Transcription In Progress", "A transcription is already running.")
            return

        self.progress_output.clear()
        self.transcript_edit.clear()

        self.update_status("Transcribing local file...", self.theme.colors.warning)
        self.transcribe_file_button.setEnabled(False)
        self.browse_file_button.setEnabled(False)
        transcription_config = self._build_transcription_config()
        grammar_config = self._build_grammar_config()
        self._append_runtime_summary(
            source="Local file",
            transcription_config=transcription_config,
            grammar_config=grammar_config,
        )

        thread = threading.Thread(
            target=self.transcribe_local_file_thread,
            args=(file_path, transcription_config, grammar_config),
            daemon=True
        )
        self.transcribe_thread = thread
        try:
            thread.start()
        except RuntimeError as e:
            self.output_queue.put(("error", f"Failed to start transcription thread: {e}"))
            self.output_queue.put(("status", "Ready"))
            self.transcribe_file_button.setEnabled(True)
            self.browse_file_button.setEnabled(True)
            return

    def transcribe_local_file_thread(
        self,
        file_path: str,
        transcription_config: TranscriptionConfig,
        grammar_config: GrammarConfig,
    ) -> None:
        """Background thread for local file transcription."""
        transcription_succeeded = False
        try:
            with _worker_queue_bridge(self.output_queue):
                self.output_queue.put(("status", "Preparing local file...", self.theme.colors.warning))
                self.output_queue.put(("progress", f"Selected file: {file_path}\n"))
                self.output_queue.put(("progress", "Inspecting local file and runtime settings...\n"))

                ffmpeg_location = find_ffmpeg()
                backend = transcription_config.batch_backend

                def _run_local_whisper() -> tuple[str, TranscriptSegments]:
                    self.output_queue.put(("progress", "Freeing GPU memory...\n"))
                    unload_gector()
                    torch_module = get_torch(context="gui_transcriber:file_empty_cache")
                    if torch_module is not None and torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()

                    (
                        reusable_execution_state,
                        requested_model_name,
                        requested_device,
                        requested_compute_type,
                    ) = TranscriberGUI._build_reusable_whisper_execution_state(self, transcription_config)
                    captured_execution_state: list[_WhisperExecutionState] = []
                    if reusable_execution_state is not None:
                        self.output_queue.put(
                            (
                                "progress",
                                "Reusing the loaded Whisper runtime for local-file transcription...\n",
                            )
                        )
                        self.output_queue.put(
                            ("status", "Preparing audio and reusing Whisper...", self.theme.colors.warning)
                        )
                    else:
                        self.output_queue.put(
                            ("status", "Preparing audio and loading Whisper...", self.theme.colors.warning)
                        )

                    try:
                        try:
                            return transcribe_local_file(
                                file_path=file_path,
                                ffmpeg_location=ffmpeg_location,
                                config=transcription_config,
                                execution_state=reusable_execution_state,
                                execution_state_observer=(
                                    captured_execution_state.append if reusable_execution_state is not None else None
                                ),
                            )
                        finally:
                            if captured_execution_state:
                                previous_device = getattr(self, "_whisper_device", "cpu")
                                TranscriberGUI._sync_loaded_whisper_execution_state(
                                    self,
                                    captured_execution_state[-1],
                                    requested_model_name=requested_model_name,
                                    requested_device=requested_device,
                                    requested_compute_type=requested_compute_type,
                                )
                                if previous_device == "cuda" and self._whisper_device == "cpu":
                                    self.output_queue.put(("gpu_status", ("cpu", "")))
                    except Exception as exc:
                        if transcription_config.device_preference != "cpu" and _needs_cuda_cpu_fallback(exc):
                            cpu_config = self._build_cpu_recovery_transcription_config(transcription_config)
                            self.output_queue.put(
                                (
                                    "progress",
                                    "CUDA ran out of memory during local-file transcription; "
                                    f"retrying full job on CPU (INT8, batch {cpu_config.batch_size})...\n",
                                )
                            )
                            self.output_queue.put(
                                ("status", "Retrying local transcription on CPU...", self.theme.colors.warning)
                            )
                            return transcribe_local_file(
                                file_path=file_path,
                                ffmpeg_location=ffmpeg_location,
                                config=cpu_config,
                            )
                        raise

                if backend in {"openai", "compare"} and not is_openai_api_configured():
                    self.output_queue.put(("error", "OPENAI_API_KEY is not set. Rotate the exposed key and set a fresh key."))
                    return

                if backend == "openai":
                    self.output_queue.put(("status", "Transcribing local file with OpenAI...", self.theme.colors.warning))
                    self.output_queue.put(("progress", "Preparing local media for OpenAI transcription...\n"))
                    transcript, segments_data = transcribe_local_file_openai(
                        file_path=file_path,
                        ffmpeg_location=ffmpeg_location,
                        config=transcription_config,
                    )
                elif backend == "compare":
                    self.output_queue.put(("status", "Comparing OpenAI and local Whisper...", self.theme.colors.warning))
                    self.output_queue.put(("progress", "Running OpenAI transcription first...\n"))
                    openai_transcript, _openai_segments = transcribe_local_file_openai(
                        file_path=file_path,
                        ffmpeg_location=ffmpeg_location,
                        config=transcription_config,
                    )
                    self.output_queue.put(("progress", "Running local Whisper transcription for comparison...\n"))
                    local_transcript, _local_segments = _run_local_whisper()
                    transcript = _format_backend_comparison(
                        openai_transcript=openai_transcript,
                        local_transcript=local_transcript,
                    )
                    segments_data = None
                else:
                    transcript, segments_data = _run_local_whisper()

                if transcript.strip():
                    grammar_enhanced = False
                    if grammar_config.enabled and transcription_config.batch_backend == "compare":
                        self.output_queue.put(("progress", "Grammar correction skipped in Compare mode to preserve raw backend outputs.\n"))
                    elif grammar_config.enabled:
                        grammar_result = _apply_optional_grammar_corrections(
                            self.output_queue,
                            transcript,
                            segments_data if segments_data else [],
                            grammar_config,
                            warning_color=self.theme.colors.warning,
                            start_status="Applying grammar corrections...",
                        )
                        if grammar_result.completed:
                            transcript = grammar_result.transcript
                            segments_data = grammar_result.segments_data
                            grammar_enhanced = grammar_result.grammar_enhanced

                    _queue_transcript_snapshot(self.output_queue, transcript, segments_data)
                    transcription_succeeded = True

                    status_msg = _build_transcription_complete_status(
                        grammar_enhanced=grammar_enhanced,
                        segment_count=len(segments_data) if segments_data else None,
                    )
                    self.output_queue.put(("status", status_msg, self.theme.colors.success_light))
                    gui_logger.info(f"\nTranscription saved. Words: {len(transcript.split())}")
                else:
                    _queue_transcript_snapshot(self.output_queue, transcript, segments_data)
                    segment_count = len(segments_data) if segments_data else 0
                    self.output_queue.put(
                        (
                            "status",
                            f"No speech detected (returned {segment_count} segments). Check the progress log for audio-track/VAD details.",
                            self.theme.colors.warning,
                        )
                    )

        except Exception as exc:
            gui_logger.error(f"Local file transcription error: {exc}", exc_info=True)
            self.output_queue.put(("error", f"Error: {exc}"))
        finally:
            self.output_queue.put(("transcribe_finished", transcription_succeeded))
            self.output_queue.put(("local_file_done", transcription_succeeded))

    def toggle_recording(self) -> None:
        """Toggle recording button state."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        """Begin live OpenAI realtime transcription."""
        if self.is_recording:
            return
        active_transcription = getattr(self, "transcribe_thread", None)
        if active_transcription and active_transcription.is_alive():
            self.append_progress(
                "Wait for the current transcription to finish before starting a new recording.\n"
            )
            self.update_status("Transcription already in progress", self.theme.colors.warning)
            return

        if not is_openai_api_configured():
            self.append_progress("OPENAI_API_KEY is not set. Rotate the exposed key and set a fresh key before using live transcription.\n")
            self.update_status("Missing OPENAI_API_KEY", self.theme.colors.error)
            return

        self._start_recording_now()

    def _resolve_requested_whisper_runtime(
        self,
        transcription_config: TranscriptionConfig,
    ) -> tuple[str, str, str]:
        requested_model_name = transcription_config.whisper_model
        requested_device, requested_compute_type = get_whisper_device_and_compute_type(
            config=transcription_config,
            verbose=False,
        )
        return requested_model_name, requested_device, requested_compute_type

    def _loaded_whisper_runtime_matches(
        self,
        *,
        requested_model_name: str,
        requested_device: str,
        requested_compute_type: str,
    ) -> bool:
        return (
            getattr(self, "whisper_model", None) is not None
            and getattr(self, "_loaded_whisper_model_name", None) == requested_model_name
            and getattr(self, "_requested_whisper_device", None) == requested_device
            and getattr(self, "_requested_compute_type", None) == requested_compute_type
        )

    def _build_reusable_whisper_execution_state(
        self,
        transcription_config: TranscriptionConfig,
    ) -> tuple[_WhisperExecutionState | None, str, str, str]:
        requested_model_name = transcription_config.whisper_model
        loaded_pipeline = getattr(self, "whisper_model", None)
        if loaded_pipeline is None:
            return None, requested_model_name, "", ""

        if getattr(self, "_loaded_whisper_model_name", None) != requested_model_name:
            return None, requested_model_name, "", ""

        (
            requested_model_name,
            requested_device,
            requested_compute_type,
        ) = TranscriberGUI._resolve_requested_whisper_runtime(self, transcription_config)
        if not TranscriberGUI._loaded_whisper_runtime_matches(
            self,
            requested_model_name=requested_model_name,
            requested_device=requested_device,
            requested_compute_type=requested_compute_type,
        ):
            return None, requested_model_name, requested_device, requested_compute_type

        loaded_device = getattr(self, "_whisper_device", "cpu")
        loaded_compute_type = getattr(self, "_loaded_compute_type", None) or (
            "float16" if loaded_device == "cuda" else "int8"
        )
        return (
            _WhisperExecutionState(
                model_name=requested_model_name,
                base_model=None,
                pipeline=loaded_pipeline,
                device=loaded_device,
                compute_type=loaded_compute_type,
            ),
            requested_model_name,
            requested_device,
            requested_compute_type,
        )

    def _sync_loaded_whisper_execution_state(
        self,
        execution_state: _WhisperExecutionState,
        *,
        requested_model_name: str,
        requested_device: str,
        requested_compute_type: str,
    ) -> None:
        self.whisper_model = execution_state.pipeline
        self._loaded_whisper_model_name = execution_state.model_name or requested_model_name
        self._requested_whisper_device = requested_device
        self._requested_compute_type = requested_compute_type
        self._whisper_device = execution_state.device
        self._loaded_compute_type = execution_state.compute_type

    def _start_recording_now(self) -> None:
        """Begin live OpenAI realtime transcription."""
        if self.is_recording:
            return

        mic_descriptor = self.mic_combo.currentText()

        if not mic_descriptor or mic_descriptor.strip() == "":
            QtWidgets.QMessageBox.warning(
                self,
                "No Microphone Selected",
                "Please select a microphone device before recording.\n\n"
                "Click the refresh button to scan for available devices."
            )
            return

        self.is_recording = True
        self._recording_stop_event.clear()  # Clear stop signal for new recording
        self._set_record_button_state(recording=True)
        self.record_button.setEnabled(True)
        self._accumulated_audio_buffer = []
        self._recording_sample_rate = OPENAI_REALTIME_INPUT_SAMPLE_RATE

        self._recording_start_time = time.time()
        self._recording_duration_seconds = 0
        self.recording_duration_label.setVisible(True)
        self.recording_timer.start()

        self.append_progress("OpenAI realtime transcription in progress. Speak now; text will stream as turns complete.\n")
        self.update_status(f"Live OpenAI transcription on: {mic_descriptor}", self.theme.colors.recording)

        thread = threading.Thread(target=self.record_audio_openai_realtime_thread, args=(mic_descriptor,), daemon=True)
        self.recording_thread = thread
        try:
            thread.start()
        except RuntimeError as e:
            self.output_queue.put(("error", f"Failed to start recording thread: {e}"))
            self.output_queue.put(("status", "Ready"))
            self._set_record_button_state(recording=False)
            self.record_button.setEnabled(True)
            self.mic_combo.setEnabled(True)
            self.is_recording = False
            self._reset_recording_ui_state()
            return

    def stop_recording(self) -> None:
        """Stop live OpenAI realtime transcription."""
        if not self.is_recording:
            return

        self._recording_stop_event.set()
        with self._audio_buffer_lock:
            self.is_recording = False

        self._set_record_button_state(recording=False)
        self._reset_recording_ui_state()
        self.update_status("Stopping OpenAI realtime transcription...", self.theme.colors.warning)
        self.append_progress("Stopping OpenAI realtime transcription and finalizing the last turn...\n")

    def load_whisper_model(
        self,
        model_name: Optional[str] = None,
        transcription_config: TranscriptionConfig | None = None,
    ) -> None:
        """Load the Whisper model used for audio transcription."""
        if hasattr(self, "gpu_memory_spin"):
            self.config.gpu_memory_fraction = self.gpu_memory_spin.value()
        active_config = transcription_config or TranscriptionConfig(**asdict(self.config.transcription))
        requested_model_name = model_name or active_config.whisper_model
        _resolved_model_name, desired_device, desired_compute_type = (
            TranscriberGUI._resolve_requested_whisper_runtime(self, active_config)
        )

        if requested_model_name != _resolved_model_name:
            active_config = TranscriptionConfig(**asdict(active_config))
            active_config.whisper_model = requested_model_name
            _resolved_model_name, desired_device, desired_compute_type = (
                TranscriberGUI._resolve_requested_whisper_runtime(self, active_config)
            )

        if TranscriberGUI._loaded_whisper_runtime_matches(
            self,
            requested_model_name=requested_model_name,
            requested_device=desired_device,
            requested_compute_type=desired_compute_type,
        ):
            self.output_queue.put(("status", "Whisper model already loaded.", self.theme.colors.success_light))
            self.output_queue.put(("model_ready", "READY"))
            return

        try:
            self.output_queue.put(("status", "Loading Whisper model...", self.theme.colors.warning))
            self.output_queue.put(("progress", f"Loading Whisper '{requested_model_name}' model (~3GB first download)...\n"))
            if self.whisper_model is not None:
                previous_device = getattr(self, "_whisper_device", "cpu")
                previous_model_name = self._loaded_whisper_model_name or "current"
                self.output_queue.put(
                    (
                        "progress",
                        "Releasing previously loaded Whisper model "
                        f"'{previous_model_name}' ({previous_device.upper()}) before loading replacement...\n",
                    )
                )
                previous_pipeline = self.whisper_model
                self.whisper_model = None
                self._loaded_whisper_model_name = None
                self._loaded_compute_type = None
                self._requested_whisper_device = None
                self._requested_compute_type = None
                self._whisper_device = "cpu"
                del previous_pipeline
                gc.collect()
                if previous_device == "cuda":
                    torch_module = get_torch(context="gui_transcriber:release_whisper_model")
                    if torch_module is not None and torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()
            if desired_device == "cuda":
                self.output_queue.put(
                    ("progress", "Freeing grammar-model GPU memory before loading Whisper...\n")
                )
                unload_gector()
                gc.collect()
                torch_module = get_torch(context="gui_transcriber:load_whisper_model")
                if torch_module is not None and torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()

            preferred_device, preferred_compute_type = desired_device, desired_compute_type
            if preferred_device == "cpu":
                if active_config.device_preference == "cpu":
                    self.output_queue.put(("progress", "CPU mode selected explicitly.\n"))
                else:
                    self.output_queue.put(("progress", "CUDA backend unavailable; using CPU...\n"))

            _base_model, pipeline, device, _compute_type = _load_whisper_pipeline_with_fallback(
                requested_model_name,
                device=preferred_device,
                compute_type=preferred_compute_type,
            )
            if preferred_device == "cuda" and device == "cpu":
                self.output_queue.put(("progress", "CUDA libraries missing; retrying on CPU...\n"))
            TranscriberGUI._sync_loaded_whisper_execution_state(
                self,
                _WhisperExecutionState(
                    model_name=requested_model_name,
                    base_model=_base_model,
                    pipeline=pipeline,
                    device=device,
                    compute_type=_compute_type,
                ),
                requested_model_name=requested_model_name,
                requested_device=desired_device,
                requested_compute_type=desired_compute_type,
            )

            if device == "cuda":
                torch_module = get_torch(context="gui_transcriber:realtime_gpu_status")
                if torch_module is not None:
                    try:
                        gpu_name = torch_module.cuda.get_device_name(0)
                        self.output_queue.put(("gpu_status", ("cuda", gpu_name)))
                    except Exception:
                        self.output_queue.put(("gpu_status", ("cuda", "CUDA GPU")))
                else:
                    self.output_queue.put(("gpu_status", ("cuda", "CTranslate2 CUDA")))
            else:
                self.output_queue.put(("gpu_status", ("cpu", "")))

            self.output_queue.put(
                ("progress", f"Whisper '{requested_model_name}' loaded on {device.upper()} ({_compute_type})\n")
            )
            self.output_queue.put(("status", f"Whisper model ready ({device.upper()})", self.theme.colors.success_light))
            self.output_queue.put(("model_ready", device.upper()))

        except Exception as exc:
            self.output_queue.put(("model_load_failed", ""))
            self.output_queue.put(("error", f"Failed to load Whisper model: {exc}"))

    def _transcribe_accumulated_audio(
        self,
        audio_buffer: list[float],
        duration_seconds: int,
        transcription_config: TranscriptionConfig,
        grammar_config: GrammarConfig,
        capture_sample_rate: int | None = None,
        _retrying: bool = False,
    ) -> None:
        """Transcribe the full accumulated audio buffer after recording stops.

        Args:
            audio_buffer: List of audio samples.
            duration_seconds: Duration of the recording in seconds.
        """
        try:
            if not self.whisper_model:
                self.output_queue.put(("error", "Whisper model not loaded!"))
                return

            if not audio_buffer or len(audio_buffer) == 0:
                self.output_queue.put(("status", "No audio to transcribe", self.theme.colors.warning))
                return

            audio_array = np.asarray(audio_buffer, dtype=np.float32)
            source_sample_rate = capture_sample_rate or self.config.recording.sample_rate
            if source_sample_rate <= 0:
                source_sample_rate = WHISPER_SAMPLE_RATE

            # Audio preprocessing (noise reduction + loudness normalization)
            if transcription_config.noise_reduction_enabled or transcription_config.normalize_audio:
                self.output_queue.put(("progress", "Preprocessing audio (noise reduction/normalization)...\n"))
                audio_array = preprocess_array(
                    audio_array,
                    sample_rate=source_sample_rate,
                    noise_reduction=transcription_config.noise_reduction_enabled,
                    normalize=transcription_config.normalize_audio,
                )

            if source_sample_rate != WHISPER_SAMPLE_RATE:
                self.output_queue.put(
                    (
                        "progress",
                        f"Resampling recorded audio from {source_sample_rate} Hz to "
                        f"{WHISPER_SAMPLE_RATE} Hz for Whisper...\n",
                    )
                )
                audio_array = _resample_audio_for_whisper(
                    audio_array,
                    source_rate=source_sample_rate,
                    target_rate=WHISPER_SAMPLE_RATE,
                )

            self.output_queue.put(("progress", f"\nTranscribing {duration_seconds}s of audio with Whisper...\n"))

            batch_size = transcription_config.batch_size
            device_for_batch = getattr(self, "_whisper_device", "cpu")
            if device_for_batch == "cpu":
                batch_size = min(batch_size, transcription_config.cpu_fallback_batch_size)
            whisper_model = self.whisper_model
            if whisper_model is None:
                self.output_queue.put(("error", "Whisper model not loaded!"))
                return

            # Use the same shared Whisper kwargs as file and YouTube transcription.
            transcribe_kwargs = _build_runtime_transcribe_kwargs(
                transcription_config,
                model_name=self._loaded_whisper_model_name or transcription_config.whisper_model,
                device=device_for_batch,
                compute_type=getattr(self, "_loaded_compute_type", None) or "int8",
                context="Recording transcription",
            )
            transcribe_kwargs["batch_size"] = min(int(transcribe_kwargs["batch_size"]), batch_size)
            candidate_duration = float(duration_seconds) if duration_seconds > 0 else None

            def _decode_segments(kwargs: dict[str, Any]) -> TranscriptSegments:
                segments, _info = whisper_model.transcribe(audio_array, **kwargs)
                collected: TranscriptSegments = []
                for segment in list(segments):
                    progress = ((segment.end / candidate_duration) * 100) if candidate_duration else 0.0
                    self.output_queue.put(
                        (
                            "progress",
                            f"Recording decode: {progress:.1f}% | "
                            f"[{segment.start:.1f}s - {segment.end:.1f}s] | "
                            f"{segment.text.strip()}\n",
                        )
                    )
                    collected.append(
                        make_transcript_segment(start=segment.start, end=segment.end, text=segment.text.strip())
                    )
                return collected

            def _normalize_candidate(
                raw_segments: TranscriptSegments,
            ) -> tuple[str, TranscriptSegments, int, int]:
                return _normalize_transcript_segments(
                    raw_segments,
                    clean_fillers=transcription_config.clean_filler_words,
                    filter_hallucinated=transcription_config.filter_hallucinations,
                    deduplicate_repetitions=transcription_config.deduplicate_repeated_segments,
                )

            segments_data = _decode_segments(transcribe_kwargs)
            final_transcript, segments_data, removed_count, hallucination_count = _normalize_candidate(
                segments_data
            )

            rejected_unreliable_transcript = False
            if transcription_result_looks_suspicious(segments_data, input_duration=candidate_duration):
                self.output_queue.put(
                    (
                        "progress",
                        "Initial microphone transcript looked suspicious; retrying with stricter anti-hallucination settings...\n",
                    )
                )
                retry_kwargs = build_suspicion_retry_kwargs(transcribe_kwargs)
                try:
                    retry_segments_data = _decode_segments(retry_kwargs)
                except RuntimeError:
                    raise
                except Exception as retry_exc:
                    self.output_queue.put(
                        (
                            "progress",
                            f"Retry failed, keeping first-pass transcript for evaluation: {retry_exc}\n",
                        )
                    )
                else:
                    final_transcript, segments_data, removed_count, hallucination_count = _normalize_candidate(
                        retry_segments_data
                    )

                if transcription_result_looks_suspicious(segments_data, input_duration=candidate_duration):
                    rejected_unreliable_transcript = True
                    final_transcript = ""
                    segments_data = []

            if removed_count > 0:
                self.output_queue.put(("progress", f"Removed {removed_count} repetitive segments\n"))

            if hallucination_count > 0:
                self.output_queue.put(("progress", f"Filtered {hallucination_count} hallucination segment(s)\n"))

            if final_transcript:

                # Grammar Post-Processing (if enabled)
                grammar_enhanced = False
                if grammar_config.enabled:
                    grammar_result = _apply_optional_grammar_corrections(
                        self.output_queue,
                        final_transcript,
                        segments_data,
                        grammar_config,
                    )
                    final_transcript = grammar_result.transcript
                    segments_data = grammar_result.segments_data
                    grammar_enhanced = grammar_result.grammar_enhanced

                word_count = len(final_transcript.split())

                _queue_transcript_snapshot(
                    self.output_queue,
                    final_transcript,
                    segments_data,
                    append=True,
                )
                self.output_queue.put(("progress", f"Transcription complete: {word_count} words\n"))

                status_msg = _build_transcription_complete_status(
                    grammar_enhanced=grammar_enhanced,
                    word_count=word_count,
                )
                self.output_queue.put(("status", status_msg, self.theme.colors.success_light))
            else:
                if hallucination_count > 0:
                    self.output_queue.put(("progress", "Only hallucinations detected - no real speech found\n"))
                    self.output_queue.put(("status", "No real speech detected (hallucinations filtered)", self.theme.colors.warning))
                elif rejected_unreliable_transcript:
                    self.output_queue.put(("progress", "Rejected unreliable microphone transcript after retry\n"))
                    self.output_queue.put(("status", "No reliable speech detected", self.theme.colors.warning))
                else:
                    self.output_queue.put(("progress", "No speech detected in recording\n"))
                    self.output_queue.put(("status", "No speech detected", self.theme.colors.warning))
                self.output_queue.put(("segments", []))

        except RuntimeError as exc:
            if self._whisper_device == "cuda" and not _retrying:
                self.output_queue.put(
                    (
                        "progress",
                        f"Whisper runtime error on CUDA ({exc}); retrying microphone transcription on CPU...\n",
                    )
                )
                try:
                    model_name = self._loaded_whisper_model_name or transcription_config.whisper_model
                    _base_model, pipeline, device, compute_type = _load_whisper_pipeline_with_fallback(
                        model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                    self.whisper_model = pipeline
                    self._whisper_device = device
                    self._loaded_compute_type = compute_type
                    self.output_queue.put(("gpu_status", ("cpu", "")))
                except Exception as reload_exc:
                    self.output_queue.put(("error", f"Transcription error: {reload_exc}"))
                    return
                self._transcribe_accumulated_audio(
                    audio_buffer,
                    duration_seconds,
                    transcription_config,
                    grammar_config,
                    capture_sample_rate,
                    _retrying=True,
                )
            else:
                self.output_queue.put(("error", f"Transcription error: {exc}"))
        except Exception as exc:
            self.output_queue.put(("error", f"Transcription error: {exc}"))
        finally:
            self.output_queue.put(("transcribe_thread_done", ""))

    def _build_openai_realtime_prompt(self, transcription_config: TranscriptionConfig) -> str | None:
        prompt_parts: list[str] = []
        if transcription_config.initial_prompt:
            prompt_parts.append(transcription_config.initial_prompt.strip())
        if transcription_config.hotwords:
            hotwords = ", ".join(term.strip() for term in transcription_config.hotwords.split(",") if term.strip())
            if hotwords:
                prompt_parts.append(f"Vocabulary and names that may appear: {hotwords}.")
        if not prompt_parts:
            return None
        return "\n".join(prompt_parts)

    def _pcm16_bytes_from_float_audio(self, audio: np.ndarray, *, sample_rate: int) -> bytes:
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono[:, 0]
        if sample_rate != OPENAI_REALTIME_INPUT_SAMPLE_RATE and mono.size:
            mono = _resample_audio_for_whisper(
                mono,
                source_rate=sample_rate,
                target_rate=OPENAI_REALTIME_INPUT_SAMPLE_RATE,
            )
        mono = np.clip(mono, -1.0, 1.0)
        return (mono * 32767.0).astype("<i2").tobytes()

    def record_audio_openai_realtime_thread(self, mic_descriptor: str) -> None:
        """Background loop streaming microphone audio to OpenAI Realtime."""
        session: OpenAIRealtimeTranscriptionSession | None = None
        receiver_thread: threading.Thread | None = None
        receiver_stop_event = threading.Event()
        receiver_finalized_event = threading.Event()
        connection_lost_event = threading.Event()
        audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=80)
        finalized_segments: TranscriptSegments = []
        partial_by_item: dict[str, str] = {}
        first_delta_at_by_item: dict[str, float] = {}
        sent_audio_bytes = 0
        started_at = time.time()

        def _elapsed() -> float:
            return max(0.0, time.time() - started_at)

        def _render_live_transcript() -> str:
            final_text = " ".join(str(segment["text"]).strip() for segment in finalized_segments if segment.get("text"))
            partial_text = " ".join(text.strip() for text in partial_by_item.values() if text.strip())
            if final_text and partial_text:
                return f"{final_text} {partial_text}"
            return final_text or partial_text

        def _queue_live_snapshot() -> None:
            self.output_queue.put(("realtime_transcript", _render_live_transcript()))
            self.output_queue.put(("segments", list(finalized_segments)))

        def _finalize_partial_items() -> bool:
            promoted = False
            end = _elapsed()
            for item_id, text in list(partial_by_item.items()):
                transcript = text.strip()
                partial_by_item.pop(item_id, None)
                start = min(first_delta_at_by_item.pop(item_id, end), end)
                if not transcript:
                    continue
                finalized_segments.append(make_transcript_segment(start=start, end=end, text=transcript))
                promoted = True
            return promoted

        def _is_remote_connection_lost(exc: Exception) -> bool:
            class_name = type(exc).__name__.lower()
            message = str(exc).lower()
            return (
                "websocketconnectionclosed" in class_name
                or "connection to remote host was lost" in message
                or "connection is already closed" in message
                or "socket is already closed" in message
            )

        def _handle_remote_connection_lost(exc: Exception) -> None:
            if connection_lost_event.is_set():
                return
            promoted = _finalize_partial_items()
            connection_lost_event.set()
            receiver_finalized_event.set()
            if finalized_segments or promoted:
                self.output_queue.put(
                    ("progress", f"OpenAI realtime connection closed; kept transcript received so far. Reconnecting... ({exc})\n")
                )
            else:
                self.output_queue.put(
                    ("progress", f"OpenAI realtime connection closed before transcript data arrived. Reconnecting... ({exc})\n")
                )
            _queue_live_snapshot()

        def _handle_realtime_event(event: dict[str, Any]) -> None:
            event_type = str(event.get("type", ""))
            if event_type == "conversation.item.input_audio_transcription.delta":
                item_id = str(event.get("item_id") or "current")
                delta = str(event.get("delta") or "")
                if not delta:
                    return
                first_delta_at_by_item.setdefault(item_id, _elapsed())
                partial_by_item[item_id] = f"{partial_by_item.get(item_id, '')}{delta}"
                _queue_live_snapshot()
            elif event_type == "conversation.item.input_audio_transcription.completed":
                item_id = str(event.get("item_id") or "current")
                transcript = str(event.get("transcript") or partial_by_item.get(item_id, "")).strip()
                partial_by_item.pop(item_id, None)
                if transcript:
                    end = _elapsed()
                    start = min(first_delta_at_by_item.pop(item_id, end), end)
                    finalized_segments.append(make_transcript_segment(start=start, end=end, text=transcript))
                    self.output_queue.put(("progress", f"OpenAI realtime turn: {transcript}\n"))
                _queue_live_snapshot()
                receiver_finalized_event.set()
            elif event_type == "error":
                error_payload = event.get("error")
                message = error_payload.get("message") if isinstance(error_payload, dict) else str(error_payload)
                self.output_queue.put(("error", f"OpenAI realtime error: {message}"))

        def _receiver_loop(active_session: OpenAIRealtimeTranscriptionSession) -> None:
            while not receiver_stop_event.is_set():
                try:
                    event = active_session.recv_event()
                except Exception as exc:
                    if not receiver_stop_event.is_set() and _is_remote_connection_lost(exc):
                        _handle_remote_connection_lost(exc)
                        return
                    if not receiver_stop_event.is_set() and not self._recording_stop_event.is_set():
                        self.output_queue.put(("error", f"OpenAI realtime receive error: {exc}"))
                    return
                if event is not None:
                    _handle_realtime_event(event)

        def _wait_for_realtime_finalization() -> bool:
            deadline = time.monotonic() + OPENAI_REALTIME_FINALIZATION_TIMEOUT_S
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return receiver_finalized_event.is_set()
                if receiver_finalized_event.wait(min(OPENAI_REALTIME_FINALIZATION_POLL_S, remaining)):
                    return True

        try:
            sd = _get_sounddevice_module()
            try:
                mic_index = int(mic_descriptor.split(":")[0]) if ":" in mic_descriptor else None
            except (ValueError, IndexError) as exc:
                self.output_queue.put(("error", f"Invalid microphone selection '{mic_descriptor}': {exc}"))
                self.output_queue.put(("recording_reset", ""))
                return

            transcription_config = self._build_transcription_config()
            realtime_prompt = self._build_openai_realtime_prompt(transcription_config)

            def _build_realtime_session() -> OpenAIRealtimeTranscriptionSession:
                return OpenAIRealtimeTranscriptionSession(
                    session_model=self.config.recording.openai_realtime_session_model,
                    transcription_model=self.config.recording.openai_realtime_transcription_model,
                    language=transcription_config.language,
                    prompt=realtime_prompt,
                )

            def _start_receiver_thread(active_session: OpenAIRealtimeTranscriptionSession) -> None:
                nonlocal receiver_thread
                receiver_thread = threading.Thread(target=_receiver_loop, args=(active_session,), daemon=True)
                receiver_thread.start()

            def _reconnect_realtime_session() -> None:
                nonlocal session, receiver_thread
                old_session = session
                if old_session is not None:
                    old_session.close()
                if receiver_thread is not None and receiver_thread.is_alive():
                    receiver_thread.join(timeout=2)
                session = _build_realtime_session()
                session.connect()
                connection_lost_event.clear()
                receiver_finalized_event.clear()
                _start_receiver_thread(session)
                self.output_queue.put(("progress", "OpenAI realtime reconnected; continuing live transcription.\n"))

            session = _build_realtime_session()
            session.connect()
            _start_receiver_thread(session)

            self.output_queue.put(("progress", f"Streaming microphone '{mic_descriptor}' to OpenAI Realtime\n"))
            self.output_queue.put(
                (
                    "progress",
                    "Realtime transcription: "
                    f"mode=transcription | model={self.config.recording.openai_realtime_transcription_model}\n",
                )
            )

            stream_sample_rate = OPENAI_REALTIME_INPUT_SAMPLE_RATE
            try:
                device_info = sd.query_devices(mic_index, "input") if mic_index is not None else sd.query_devices(kind="input")
                default_rate = int(device_info.get("default_samplerate", stream_sample_rate))
                if default_rate > 0:
                    stream_sample_rate = default_rate
            except Exception:
                stream_sample_rate = OPENAI_REALTIME_INPUT_SAMPLE_RATE

            self._recording_sample_rate = stream_sample_rate

            def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
                if status:
                    self.output_queue.put(("progress", f"Audio status: {status}\n"))
                if self._recording_stop_event.is_set():
                    raise sd.CallbackStop()
                pcm_bytes = self._pcm16_bytes_from_float_audio(indata, sample_rate=stream_sample_rate)
                try:
                    audio_queue.put_nowait(pcm_bytes)
                except queue.Full:
                    self.output_queue.put(("progress", "OpenAI realtime audio buffer is full; dropping a small chunk.\n"))

            with sd.InputStream(device=mic_index, channels=1, samplerate=stream_sample_rate, callback=audio_callback):
                while not self._recording_stop_event.is_set():
                    if connection_lost_event.is_set():
                        _reconnect_realtime_session()
                        continue
                    try:
                        pcm_bytes = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    try:
                        session.append_pcm16(pcm_bytes)
                    except Exception as exc:
                        if _is_remote_connection_lost(exc):
                            _handle_remote_connection_lost(exc)
                            continue
                        raise
                    sent_audio_bytes += len(pcm_bytes)

                while not connection_lost_event.is_set() and not audio_queue.empty():
                    pcm_bytes = audio_queue.get_nowait()
                    try:
                        session.append_pcm16(pcm_bytes)
                    except Exception as exc:
                        if _is_remote_connection_lost(exc):
                            _handle_remote_connection_lost(exc)
                            break
                        raise
                    sent_audio_bytes += len(pcm_bytes)

            if connection_lost_event.is_set():
                _queue_live_snapshot()
            elif sent_audio_bytes > 0:
                if not _wait_for_realtime_finalization():
                    _finalize_partial_items()
                    self.output_queue.put(("progress", "Timed out waiting for OpenAI realtime final transcription.\n"))
            else:
                self.output_queue.put(("progress", "OpenAI realtime recording ended before enough audio was captured.\n"))
            _queue_live_snapshot()
            self.output_queue.put(("status", "OpenAI realtime transcription stopped", self.theme.colors.success_light))
        except Exception as exc:
            self.output_queue.put(("error", f"OpenAI realtime recording error: {exc}"))
            self.output_queue.put(("recording_reset", ""))
        finally:
            receiver_stop_event.set()
            if session is not None:
                session.close()
            if receiver_thread is not None and receiver_thread.is_alive():
                receiver_thread.join(timeout=2)
            self.output_queue.put(("recording_thread_done", ""))

    def record_audio_whisper_thread(self, mic_descriptor: str) -> None:
        """Background loop capturing audio."""
        try:
            sd = _get_sounddevice_module()
            mic_index: Optional[int]
            try:
                mic_index = int(mic_descriptor.split(":")[0]) if ":" in mic_descriptor else None
            except (ValueError, IndexError) as exc:
                self.output_queue.put(("error", f"Invalid microphone selection '{mic_descriptor}': {exc}"))
                self.output_queue.put(("recording_reset", ""))
                return

            self.output_queue.put(("progress", f"Recording audio from '{mic_descriptor}'\n"))

            # Validate microphone is still available before starting stream
            if mic_index is not None:
                try:
                    devices = sd.query_devices()
                    if mic_index >= len(devices):
                        raise ValueError(f"Microphone index {mic_index} no longer valid (device disconnected?)")
                    device_info = devices[mic_index]
                    if device_info.get('max_input_channels', 0) < 1:
                        raise ValueError(f"Device '{device_info.get('name', mic_index)}' has no input channels")
                except sd.PortAudioError as e:
                    self.output_queue.put(("error", f"Audio system error: {e}"))
                    self.output_queue.put(("recording_reset", ""))
                    return
                except ValueError as e:
                    self.output_queue.put(("error", f"Microphone unavailable: {e}"))
                    self.output_queue.put(("recording_reset", ""))
                    return

            # Use 16kHz sample rate (Whisper's native rate) when supported by the device
            sample_rate = WHISPER_SAMPLE_RATE
            if mic_index is not None:
                try:
                    device_info = sd.query_devices(mic_index)
                    device_rate = int(device_info.get("default_samplerate", sample_rate))
                    # If device doesn't list 16kHz as default, fall back to its default rate
                    if device_rate and device_rate != sample_rate:
                        sample_rate = device_rate
                        self.output_queue.put(("progress", f"Using device sample rate: {sample_rate} Hz\n"))
                except Exception:
                    # If query fails, stick with 16kHz and let PortAudio surface errors if unsupported
                    pass
            self._recording_sample_rate = sample_rate
            temp_buffer: list[float] = []

            def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
                if status:
                    self.output_queue.put(("progress", f"Audio status: {status}\n"))
                if self._recording_stop_event.is_set():
                    raise sd.CallbackStop()
                with self._audio_buffer_lock:
                    samples = cast(list[float], indata[:, 0].astype(np.float32).tolist())
                    temp_buffer.extend(samples)

            with sd.InputStream(device=mic_index, channels=1, samplerate=sample_rate, callback=audio_callback):
                last_transfer_time = time.time()

                # Use atomic Event instead of boolean flag for thread-safe stop detection
                while not self._recording_stop_event.is_set():
                    current_time = time.time()
                    if current_time - last_transfer_time >= 1.0:
                        with self._audio_buffer_lock:
                            if temp_buffer:
                                self._accumulated_audio_buffer.extend(temp_buffer)
                                buffer_len = len(self._accumulated_audio_buffer)
                                temp_buffer.clear()
                                last_transfer_time = current_time

                                duration = buffer_len / sample_rate
                                if int(duration) % 10 == 0 and int(duration) > 0:
                                    self.output_queue.put(("progress", f"Captured {int(duration)}s of audio...\n"))

                    # Wait with timeout, using the stop event for efficient cancellation
                    self._recording_stop_event.wait(0.1)

                # Final transfer of any remaining samples
                with self._audio_buffer_lock:
                    if temp_buffer:
                        self._accumulated_audio_buffer.extend(temp_buffer)
                        temp_buffer.clear()

        except Exception as exc:
            self.output_queue.put(("error", f"Recording error: {exc}"))
            self.output_queue.put(("recording_reset", ""))
            # Clear accumulated buffer to prevent stale data in next recording
            with self._audio_buffer_lock:
                self._accumulated_audio_buffer.clear()
        finally:
            self.output_queue.put(("recording_thread_done", ""))

    def process_queue(self) -> None:
        """Process messages coming from worker threads."""
        while True:
            try:
                msg = self.output_queue.get_nowait()
            except queue.Empty:
                break

            kind = msg[0]
            if kind == "status":
                message = msg[1]
                color = msg[2] if len(msg) > 2 and msg[2] else self.theme.colors.success_light
                self.update_status(message, color)
            elif kind == "progress":
                self.append_progress(msg[1])
            elif kind == "transcript":
                self.update_transcript(msg[1])
            elif kind == "realtime_transcript":
                self.update_transcript(msg[1])
            elif kind == "append_transcript":
                new_text = msg[1]
                current_text = self.transcript_edit.toPlainText()
                separator = "\n\n" if current_text else ""
                combined = f"{current_text}{separator}{new_text}"
                self.update_transcript(combined)
            elif kind == "segments":
                segments_data = cast(Optional[TranscriptSegments], msg[1])
                self._current_segments_data = segments_data if segments_data else None
            elif kind == "cancelled":
                self.update_status("Transcription cancelled", self.theme.colors.warning)
                self.transcribe_button.setEnabled(True)
                self.cancel_youtube_button.setVisible(False)
                self.cancel_youtube_button.setEnabled(True)
            elif kind == "startup_missing_dependencies":
                missing = list(cast(Sequence[str], msg[1]))
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing Dependencies",
                    _format_missing_dependencies_message(missing),
                )
                self.update_status("Some dependencies are missing", self.theme.colors.warning)
            elif kind == "startup_dependency_check_failed":
                gui_logger.warning("Startup dependency warning could not be completed: %s", msg[1])
            elif kind == "error":
                QtWidgets.QMessageBox.critical(self, "Error", msg[1])
                self.update_status("An error occurred", self.theme.colors.error)
                self.transcribe_button.setEnabled(True)
                self.cancel_youtube_button.setVisible(False)
                self.cancel_youtube_button.setEnabled(True)
                # Flash error on YouTube card
                if hasattr(self, '_youtube_card'):
                    self._flash_card_error(self._youtube_card)
            elif kind == "recording_reset":
                if self.is_recording:
                    self.is_recording = False
                    self._set_record_button_state(recording=False)
                self._reset_recording_ui_state()
            elif kind == "transcribe_finished":
                success = bool(msg[1]) if len(msg) > 1 else False
                self.transcribe_button.setEnabled(True)
                self.cancel_youtube_button.setVisible(False)
                self.cancel_youtube_button.setEnabled(True)
                if success and hasattr(self, '_transcript_card'):
                    self._flash_card_success(self._transcript_card)
            elif kind == "transcribe_thread_done":
                self.transcribe_thread = None
            elif kind == "local_file_done":
                success = bool(msg[1]) if len(msg) > 1 else False
                self.transcribe_file_button.setEnabled(True)
                self.browse_file_button.setEnabled(True)
                self.transcribe_thread = None
                if success and hasattr(self, '_transcript_card'):
                    self._flash_card_success(self._transcript_card)
            elif kind == "recording_thread_done":
                self.recording_thread = None
            elif kind == "model_ready":
                self._loading_model = False
                self.record_button.setEnabled(True)
                if self._pending_record_start and not self.is_recording:
                    self._pending_record_start = False
                    self._start_recording_now()
            elif kind == "gpu_status":
                device_type = msg[1][0]
                gpu_name = msg[1][1] if len(msg[1]) > 1 else ""
                if device_type == "cuda":
                    if len(gpu_name) > 30:
                        gpu_name = gpu_name[:27] + "..."
                    self.gpu_status_label.setText(f"GPU: {gpu_name}")
                    self.gpu_status_label.setStyleSheet(self.theme.get_gpu_status_style(has_gpu=True))
                else:
                    self.gpu_status_label.setText("CPU Mode")
                    self.gpu_status_label.setStyleSheet(self.theme.get_gpu_status_style(has_gpu=False))
            elif kind == "gpu_status_unknown":
                self.gpu_status_label.setText("GPU: Unknown")
                self.gpu_status_label.setStyleSheet("color: #999; font-weight: 500; padding: 0 10px;")
            elif kind == "model_load_failed":
                self._loading_model = False
                self.record_button.setEnabled(True)
                self._pending_record_start = False

    def handle_clear(self) -> None:
        """Clear all text outputs."""
        if self.is_recording:
            QtWidgets.QMessageBox.information(
                self, "Recording in progress", "Stop the recording before clearing the transcript."
            )
            return

        self.transcript_edit.clear()
        self.progress_output.clear()
        self._current_segments_data = None
        self.update_status("Cleared", self.theme.colors.success_light)

    def handle_save(self) -> None:
        """Persist transcript to a file with format selection."""
        content = self.transcript_edit.toPlainText().strip()
        if not content:
            QtWidgets.QMessageBox.warning(self, "No Transcript", "There is no transcript to save.")
            return

        filename, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save transcript as",
            os.getcwd(),
            "Text Files (*.txt);;Subtitle Files (*.srt);;JSON Files (*.json);;All Files (*.*)",
        )

        if filename:
            if os.path.exists(filename):
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "File Exists",
                    f"The file '{os.path.basename(filename)}' already exists.\n\nDo you want to overwrite it?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )
                if reply == QtWidgets.QMessageBox.StandardButton.No:
                    return

            try:
                if "*.srt" in selected_filter or filename.endswith(".srt"):
                    self._save_as_srt(filename, content)
                elif "*.json" in selected_filter or filename.endswith(".json"):
                    self._save_as_json(filename, content)
                else:
                    self._save_as_txt(filename, content)

                QtWidgets.QMessageBox.information(self, "Saved", f"Transcript saved to:\n{filename}")
                self.update_status(f"Saved to {os.path.basename(filename)}", self.theme.colors.success_light)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save file: {exc}")

    def _save_as_txt(self, filename: str, content: str) -> None:
        """Save as plain text file."""
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _save_as_srt(self, filename: str, content: str) -> None:
        """Save as SRT subtitle file."""
        with open(filename, "w", encoding="utf-8") as handle:
            if self._current_segments_data:
                srt_content = format_transcript_as_srt(self._current_segments_data)
                handle.write(srt_content)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                for idx, sentence in enumerate(sentences, start=1):
                    if not sentence.strip():
                        continue
                    start_seconds = (idx - 1) * 3
                    end_seconds = idx * 3
                    start_time = format_srt_timestamp(start_seconds)
                    end_time = format_srt_timestamp(end_seconds)
                    handle.write(f"{idx}\n{start_time} --> {end_time}\n{sentence.strip()}\n\n")

    def _save_as_json(self, filename: str, content: str) -> None:
        """Save as JSON file."""
        if self._current_segments_data:
            json_content = format_transcript_as_json(self._current_segments_data)
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(json_content)
            return

        sentences = re.split(r'(?<=[.!?])\s+', content)
        segments = [
            {
                "id": idx,
                "text": sentence.strip(),
                "start_time": (idx - 1) * 3,
                "end_time": idx * 3,
                "duration": 3,
                "estimated": True,
            }
            for idx, sentence in enumerate(sentences, start=1)
            if sentence.strip()
        ]

        data = {
            "transcript": content,
            "segments": segments,
            "metadata": {
                "format": "json",
                "generated_by": "Speech Transcriber",
                "character_count": len(content),
                "segment_count": len(segments),
                "has_real_timestamps": False,
            },
        }

        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def handle_copy(self) -> None:
        """Copy transcript contents to clipboard."""
        content = self.transcript_edit.toPlainText().strip()
        if not content:
            QtWidgets.QMessageBox.warning(self, "No Transcript", "There is no transcript to copy.")
            return

        clipboard = QtGui.QGuiApplication.clipboard()
        if clipboard is None:
            QtWidgets.QMessageBox.warning(self, "Clipboard Unavailable", "Could not access the system clipboard.")
            return

        clipboard.setText(content)
        self.update_status("Transcript copied to clipboard", self.theme.colors.success_light)

    def closeEvent(self, a0: QtGui.QCloseEvent | None) -> None:
        """Ensure background threads stop cleanly when window closes."""
        self._save_settings()

        if self.is_recording:
            self.is_recording = False
            self._set_record_button_state(recording=False)
        self._recording_stop_event.set()
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2)

        if self.transcribe_thread and self.transcribe_thread.is_alive():
            self._youtube_cancel_event.set()
            gui_logger.info("Waiting for active transcription thread to finish on exit...")
            self.transcribe_thread.join(timeout=3)

            if self.transcribe_thread.is_alive():
                gui_logger.warning("Active transcription thread did not stop cleanly")

        close_event = a0 or QtGui.QCloseEvent()
        super().closeEvent(close_event)

    def run(self) -> None:
        """Show the window."""
        self.show()

    def _flash_card_success(self, card: GlassCard) -> None:
        """Flash green glow on a card to indicate success.

        Args:
            card: Card to flash.
        """
        if hasattr(card, 'flash_glow'):
            card.flash_glow("success", 600)

    def _flash_card_error(self, card: GlassCard) -> None:
        """Flash red glow on a card to indicate error.

        Args:
            card: Card to flash.
        """
        if hasattr(card, 'flash_glow'):
            card.flash_glow("error", 600)


def main() -> None:
    """Entry point for the PyQt application."""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Speech Transcriber")
    gui = TranscriberGUI()
    gui.run()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "Fatal Error", f"Application error: {exc}")
        sys.exit(1)
