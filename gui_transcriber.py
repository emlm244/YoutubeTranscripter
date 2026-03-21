"""
YouTube Transcriber - Modern GUI Application with PyQt6.

Provides YouTube transcription and high-quality microphone recording with batch
speech-to-text processing using Whisper AI.

Features Material Design styling, responsive layout, and theme system.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
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
    _normalize_transcript_segments,
    check_dependencies,
    download_audio,
    extract_video_id,
    find_ffmpeg,
    format_srt_timestamp,
    format_transcript_with_timestamps,
    format_transcript_as_srt,
    format_transcript_as_json,
    get_youtube_transcript,
    transcribe_audio,
    transcribe_local_file,
    validate_youtube_url,
    get_whisper_cuda_status,
    get_whisper_device_and_compute_type,
)

# Import new modules
from config import (
    ACCURACY_PRESETS,
    DEFAULT_PRESET,
    GrammarConfig,
    TranscriptionConfig,
    apply_preset,
    get_config,
    save_config,
)
from grammar_postprocessor import check_grammar_status, post_process_grammar, unload_gector
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


QueueMessage = tuple[Any, ...]
LoggerState = tuple[logging.Logger, int]

BACKEND_LOGGER_LEVELS: dict[str, int] = {
    "youtube_transcriber": logging.INFO,
    "grammar_postprocessor": logging.INFO,
    "faster_whisper": logging.INFO,
}
BACKEND_LOG_NAME = "youtube_transcriber.log"

_backend_file_handler: RotatingFileHandler | None = None


def _get_sounddevice_module():
    """Import sounddevice only when microphone paths need it."""
    import sounddevice

    return sounddevice


@dataclass(frozen=True)
class GrammarPassResult:
    """Outcome of an optional grammar-processing pass."""

    transcript: str
    segments_data: TranscriptSegments
    grammar_enhanced: bool
    completed: bool


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
    """PyQt6-based GUI for YouTube transcription and microphone recording with Whisper AI."""

    def __init__(self) -> None:
        super().__init__()

        # Initialize theme manager
        self.theme = get_theme_manager()
        self.config = get_config()
        self._whisper_device = "cpu"

        self.setWindowTitle("YouTube Transcriber")
        self._configure_window_size()

        self.output_queue: queue.Queue[QueueMessage] = queue.Queue()
        self.is_recording = False
        self.whisper_model: Optional[Any] = None  # WhisperModel loaded lazily
        self.recording_thread: Optional[threading.Thread] = None
        self.transcribe_thread: Optional[threading.Thread] = None
        self._loading_model = False
        self._pending_record_start = False
        self._realtime_model_name = "distil-large-v3"

        # Cancellation support
        self._youtube_cancel_event = threading.Event()
        self._recording_stop_event = threading.Event()

        # Recording timer
        self._recording_start_time: Optional[float] = None
        self._recording_duration_seconds = 0
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

        self.populate_microphones()
        self.update_status("Ready")
        gui_logger.info("YouTube Transcriber GUI initialized")

        self._load_settings()
        self._check_dependencies_on_startup()

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

        self.grammar_enhance_checkbox.setChecked(self.config.grammar.enabled)

        last_preset = self.config.ui.transcription_preset or DEFAULT_PRESET
        preset_idx = self.preset_combo.findData(last_preset)
        if preset_idx >= 0:
            self.preset_combo.setCurrentIndex(preset_idx)

        last_hotwords = self.config.transcription.hotwords or ""
        if last_hotwords:
            self.hotwords_input.setText(last_hotwords)

        self.filler_cleanup_checkbox.setChecked(self.config.transcription.clean_filler_words)
        self.noise_reduction_checkbox.setChecked(self.config.transcription.noise_reduction_enabled)
        self.normalize_audio_checkbox.setChecked(self.config.transcription.normalize_audio)

        gui_logger.debug("Settings loaded from previous session")

    def _save_settings(self) -> None:
        """Save settings for next session."""
        settings = QtCore.QSettings("AnthropicClaude", "YouTubeTranscriber")

        settings.setValue("splitter/state", self.content_splitter.saveState())
        self.config.ui.last_youtube_url = self.url_input.text().strip()
        self.config.ui.output_format = str(self.format_combo.currentData() or "plain")
        self.config.recording.default_microphone = self.mic_combo.currentText()
        self.config.grammar.enabled = self.grammar_enhance_checkbox.isChecked()
        self.config.ui.transcription_preset = str(self.preset_combo.currentData() or DEFAULT_PRESET)
        hotwords = self.hotwords_input.text().strip()
        self.config.transcription.hotwords = hotwords or None
        self.config.transcription.clean_filler_words = self.filler_cleanup_checkbox.isChecked()
        self.config.transcription.noise_reduction_enabled = self.noise_reduction_checkbox.isChecked()
        self.config.transcription.normalize_audio = self.normalize_audio_checkbox.isChecked()
        save_config()

        gui_logger.debug("Settings saved")

    def _check_dependencies_on_startup(self) -> None:
        """Check for missing dependencies and warn the user."""
        all_ok, missing = check_dependencies()

        if not all_ok:
            missing_list = "\n".join(f"  - {m}" for m in missing)
            message = (
                f"Missing Dependencies Detected:\n\n"
                f"{missing_list}\n\n"
                f"Some features may not work correctly.\n\n"
                f"To install missing dependencies:\n"
                f"1. Open a terminal/command prompt\n"
                f"2. Run: pip install -r requirements.txt\n"
                f"3. For FFmpeg: winget install FFmpeg (Windows)"
            )

            QtWidgets.QMessageBox.warning(self, "Missing Dependencies", message)
            gui_logger.warning(f"Missing dependencies: {', '.join(missing)}")

        self._detect_gpu_on_startup()

    def _detect_gpu_on_startup(self) -> None:
        """Detect GPU availability and update status label on startup."""
        try:
            whisper_cuda_ok, gpu_name = get_whisper_cuda_status()
            if whisper_cuda_ok:
                if not gpu_name:
                    gpu_name = "CTranslate2 CUDA"
                if len(gpu_name) > 30:
                    gpu_name = gpu_name[:27] + "..."
                self.gpu_status_label.setText(f"GPU: {gpu_name}")
                self.gpu_status_label.setStyleSheet(self.theme.get_gpu_status_style(True))
                gui_logger.info(f"Whisper CUDA backend detected: {gpu_name}")
            else:
                self.gpu_status_label.setText("CPU Mode")
                self.gpu_status_label.setStyleSheet(self.theme.get_gpu_status_style(False))
                gui_logger.info("Whisper CUDA backend not detected - using CPU mode")
        except Exception as e:
            self.gpu_status_label.setText("GPU: Unknown")
            self.gpu_status_label.setStyleSheet("color: #999; font-weight: 500; padding: 0 10px;")
            gui_logger.warning(f"Failed to detect GPU: {e}")

    def _build_ui(self) -> None:
        """Build the main UI with glassmorphic components and animations."""
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(6)

        # Title section - compact
        title = QtWidgets.QLabel("YouTube Transcriber & Speech-to-Text Recording")
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
        self._live_preview_card = self._build_live_preview_card()
        self._progress_card = self._build_progress_card()
        self._transcript_card = self._build_transcript_card()

        self.content_splitter.addWidget(self._live_preview_card)
        self.content_splitter.addWidget(self._progress_card)
        self.content_splitter.addWidget(self._transcript_card)

        # Restore splitter state or set defaults
        settings = QtCore.QSettings("AnthropicClaude", "YouTubeTranscriber")
        splitter_state = settings.value("splitter/state")
        if splitter_state:
            self.content_splitter.restoreState(splitter_state)
        else:
            self.content_splitter.setDefaultSizes([0.40, 0.20, 0.40])

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
        self.gpu_status_label.setStyleSheet(self.theme.get_gpu_status_style(False))

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
        status_bar.addWidget(self.recording_duration_label)
        self.setStatusBar(status_bar)

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
        self.record_status_label = QtWidgets.QLabel("Batch transcription with Whisper AI")
        self.record_status_label.setStyleSheet(self.theme.get_recording_status_style(False))

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

        # Row 1: Accuracy preset + hotwords
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(10)

        preset_label = QtWidgets.QLabel("Preset:")
        self.preset_combo = QtWidgets.QComboBox()
        for key, preset in ACCURACY_PRESETS.items():
            self.preset_combo.addItem(preset.name, key)
        default_idx = self.preset_combo.findData(DEFAULT_PRESET)
        if default_idx >= 0:
            self.preset_combo.setCurrentIndex(default_idx)
        self.preset_combo.setToolTip(
            "Speed: distil-large-v3, fastest\n"
            "Balanced: distil-large-v3, accent-aware prompt, language=en\n"
            "Maximum Accuracy: large-v3, accent-aware prompt, language=en"
        )
        self.preset_combo.setFixedWidth(180)

        hotwords_label = QtWidgets.QLabel("Hotwords:")
        self.hotwords_input = QtWidgets.QLineEdit()
        self.hotwords_input.setPlaceholderText("Optional: meeting terms, names, jargon")
        self.hotwords_input.setToolTip(
            "Domain-specific vocabulary to help Whisper recognize.\n"
            "Enter names, acronyms, or technical terms separated by commas.\n"
            "Example: Anthropic, Claude, RLHF, transformer"
        )

        row1.addWidget(preset_label)
        row1.addWidget(self.preset_combo)
        row1.addWidget(hotwords_label)
        row1.addWidget(self.hotwords_input, stretch=1)

        # Row 2: Checkboxes
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(10)

        # Grammar Enhancement checkbox
        self.grammar_enhance_checkbox = QtWidgets.QCheckBox("Grammar Fix")
        self.grammar_enhance_checkbox.setChecked(True)
        self.grammar_enhance_checkbox.setToolTip(
            "Grammar & spelling correction after transcription.\n\n"
            "Primary: GECToR (GPU-accelerated, high accuracy)\n"
            "Fallback: LanguageTool (local, no API key)\n\n"
            "Both run locally - no cloud dependencies."
        )

        # Grammar status indicator
        self.grammar_status_label = QtWidgets.QLabel("")
        self.grammar_status_label.setStyleSheet(f"color: {self.theme.colors.text_secondary}; font-size: 11px;")
        self._update_grammar_status()

        # Filler word cleanup checkbox
        self.filler_cleanup_checkbox = QtWidgets.QCheckBox("Clean Fillers")
        self.filler_cleanup_checkbox.setChecked(True)
        self.filler_cleanup_checkbox.setToolTip(
            "Remove filler words (umm, uh, you know, I mean, etc.)\n"
            "from the transcription output."
        )

        # Noise reduction checkbox
        self.noise_reduction_checkbox = QtWidgets.QCheckBox("Noise Reduction")
        self.noise_reduction_checkbox.setChecked(True)
        self.noise_reduction_checkbox.setToolTip(
            "Apply spectral noise reduction before transcription.\n"
            "Useful for hiss, fan noise, and room noise."
        )

        self.normalize_audio_checkbox = QtWidgets.QCheckBox("Normalize Audio")
        self.normalize_audio_checkbox.setChecked(True)
        self.normalize_audio_checkbox.setToolTip(
            "Apply loudness normalization before transcription.\n"
            "Useful when speech is too quiet or inconsistent."
        )

        row2.addWidget(self.grammar_enhance_checkbox)
        row2.addWidget(self.grammar_status_label)
        row2.addWidget(self.filler_cleanup_checkbox)
        row2.addWidget(self.noise_reduction_checkbox)
        row2.addWidget(self.normalize_audio_checkbox)
        row2.addStretch(1)

        main_layout.addLayout(row1)
        main_layout.addLayout(row2)

        card.addLayout(main_layout)
        card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)
        return card

    def _update_grammar_status(self) -> None:
        """Update the grammar status indicator."""
        if not self.grammar_enhance_checkbox.isChecked():
            self.grammar_status_label.setText("(Disabled)")
            self.grammar_status_label.setStyleSheet(
                f"color: {self.theme.colors.text_secondary}; font-size: 11px;"
            )
            return

        is_available, status = check_grammar_status(lazy=True)
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

    def _build_live_preview_card(self) -> MaterialCard:
        """Build the live preview card."""
        card = MaterialCard("Live Preview", self, elevation=1)

        self.live_preview = QtWidgets.QTextEdit()
        self.live_preview.setObjectName("LivePreview")
        self.live_preview.setPlaceholderText("Recording status appears here...")
        self.live_preview.setReadOnly(True)
        self.live_preview.setMinimumHeight(60)
        self.live_preview.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        card.addWidget(self.live_preview, stretch=1)
        return card

    def _build_progress_card(self) -> MaterialCard:
        """Build the progress output card."""
        card = MaterialCard("Progress", self, elevation=1)

        self.progress_output = QtWidgets.QPlainTextEdit()
        self.progress_output.setObjectName("ProgressOutput")
        self.progress_output.setReadOnly(True)
        self.progress_output.setMinimumHeight(50)
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
            self.record_status_label.setStyleSheet(self.theme.get_recording_status_style(True))
        else:
            self.record_button.setText("Start Recording")
            self.record_button.setVariant("success")
            self.record_button.setToolTip("Start recording speech-to-text (Ctrl+R)")
            self.record_status_label.setText("High-quality batch transcription with Whisper AI")
            self.record_status_label.setStyleSheet(self.theme.get_recording_status_style(False))

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
        self.grammar_enhance_checkbox.stateChanged.connect(self._update_grammar_status)

        # Keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self).activated.connect(self.toggle_recording)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self).activated.connect(self.handle_save)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+C"), self).activated.connect(self.handle_copy)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self).activated.connect(self.handle_clear)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+T"), self).activated.connect(self.start_youtube_transcription)

    def populate_microphones(self) -> None:
        """Scan for available microphones and update the combo box."""
        current_selection = self.mic_combo.currentText()
        microphones = self.get_microphone_list()

        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        self.mic_combo.addItems(microphones)
        self.mic_combo.blockSignals(False)

        if current_selection and current_selection in microphones:
            index = microphones.index(current_selection)
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
            except Exception:
                pass

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

    def append_progress(self, message: str) -> None:
        """Append a progress message with memory limit."""
        if not message:
            return

        self.progress_output.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.progress_output.insertPlainText(message)

        MAX_LOG_LINES = 5000
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

    def update_live_preview(self, text: str) -> None:
        """Replace live preview text."""
        self.live_preview.setPlainText(text)
        scrollbar = self.live_preview.verticalScrollBar()
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

    def _build_transcription_config(self) -> TranscriptionConfig:
        """Build TranscriptionConfig from current GUI settings."""
        config = TranscriptionConfig(**asdict(self.config.transcription))
        preset_key = self.preset_combo.currentData() or DEFAULT_PRESET
        apply_preset(config, preset_key)

        # Override with GUI-specific settings
        hotwords = self.hotwords_input.text().strip()
        if hotwords:
            config.hotwords = hotwords
        config.clean_filler_words = self.filler_cleanup_checkbox.isChecked()
        config.noise_reduction_enabled = self.noise_reduction_checkbox.isChecked()
        config.normalize_audio = self.normalize_audio_checkbox.isChecked()
        return config

    def _build_grammar_config(self) -> GrammarConfig:
        """Build GrammarConfig from current GUI settings."""
        config = GrammarConfig(**asdict(self.config.grammar))
        config.enabled = self.grammar_enhance_checkbox.isChecked()
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

        try:
            with _worker_queue_bridge(self.output_queue):
                self.output_queue.put(("status", "Validating URL...", self.theme.colors.warning))

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

                    self.output_queue.put(("status", "Loading Whisper and transcribing audio...", self.theme.colors.warning))

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
                    if grammar_config.enabled:
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
            self.output_queue.put(("transcribe_finished", ""))
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
        try:
            with _worker_queue_bridge(self.output_queue):
                self.output_queue.put(("status", "Preparing local file...", self.theme.colors.warning))
                self.output_queue.put(("progress", f"Selected file: {file_path}\n"))

                # Unload GECToR model to free GPU memory for Whisper
                self.output_queue.put(("progress", "Freeing GPU memory...\n"))
                unload_gector()
                torch_module = get_torch(context="gui_transcriber:file_empty_cache")
                if torch_module is not None and torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()

                ffmpeg_location = find_ffmpeg()
                self.output_queue.put(("status", "Analyzing audio tracks and loading Whisper...", self.theme.colors.warning))

                transcript, segments_data = transcribe_local_file(
                    file_path=file_path,
                    ffmpeg_location=ffmpeg_location,
                    config=transcription_config,
                )

                if transcript.strip():
                    grammar_enhanced = False
                    if grammar_config.enabled:
                        grammar_result = _apply_optional_grammar_corrections(
                            self.output_queue,
                            transcript,
                            segments_data,
                            grammar_config,
                            warning_color=self.theme.colors.warning,
                            start_status="Applying grammar corrections...",
                        )
                        if grammar_result.completed:
                            transcript = grammar_result.transcript
                            segments_data = grammar_result.segments_data
                            grammar_enhanced = grammar_result.grammar_enhanced

                    _queue_transcript_snapshot(self.output_queue, transcript, segments_data)

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
            self.output_queue.put(("transcribe_finished", ""))
            self.output_queue.put(("local_file_done", ""))

    def toggle_recording(self) -> None:
        """Toggle recording button state."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        """Begin audio recording for later transcription."""
        if self.is_recording:
            return

        if self.whisper_model is None:
            if not self._loading_model:
                self._loading_model = True
                self.record_button.setEnabled(False)
                self.live_preview.clear()
                self.live_preview.setPlainText(
                    "Loading Whisper AI model (distil-large-v3)...\n\n"
                    "This may take 30-60 seconds on first run.\n"
                    "The model will be downloaded if not cached.\n\n"
                    "Please wait - recording will start automatically when ready."
                )
                threading.Thread(target=self.load_whisper_model, daemon=True).start()
            self._pending_record_start = True
            self.update_status("Loading Whisper model for transcription...", self.theme.colors.warning)
            return

        self._start_recording_now()

    def _start_recording_now(self) -> None:
        """Begin recording assuming the model is ready."""
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

        self._recording_start_time = time.time()
        self._recording_duration_seconds = 0
        self.recording_duration_label.setVisible(True)
        self.recording_timer.start()

        self.update_live_preview("Recording in progress... Speak now! Click Stop when finished.")
        self.update_status(f"Recording on: {mic_descriptor}", self.theme.colors.recording)

        thread = threading.Thread(target=self.record_audio_whisper_thread, args=(mic_descriptor,), daemon=True)
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
            self.recording_timer.stop()
            return

    def stop_recording(self) -> None:
        """Stop audio capture and trigger transcription."""
        if not self.is_recording:
            return

        # Signal the recording thread to stop using atomic Event
        self._recording_stop_event.set()

        # Acquire lock to safely copy the buffer and update state
        with self._audio_buffer_lock:
            self.is_recording = False
            audio_buffer_copy = self._accumulated_audio_buffer.copy()

        self._set_record_button_state(recording=False)

        self.recording_timer.stop()
        self._recording_start_time = None
        self.recording_duration_label.setVisible(False)

        duration = self._recording_duration_seconds

        if audio_buffer_copy and len(audio_buffer_copy) > 0:
            self.update_live_preview("Processing your recording... This may take a moment.")
            self.update_status(f"Transcribing {duration}s of audio...", self.theme.colors.warning)
            transcription_config = self._build_transcription_config()
            grammar_config = self._build_grammar_config()

            thread = threading.Thread(
                target=self._transcribe_accumulated_audio,
                args=(audio_buffer_copy, duration, transcription_config, grammar_config),
                daemon=True
            )
            self.transcribe_thread = thread
            try:
                thread.start()
            except RuntimeError as e:
                self.transcribe_thread = None
                self.output_queue.put(("error", f"Failed to start transcription thread: {e}"))
                self.output_queue.put(("status", "Ready"))
                self.update_live_preview("")
        else:
            self.update_status("Recording stopped (no audio captured)", self.theme.colors.warning)
            self.update_live_preview("")

    def load_whisper_model(self) -> None:
        """Load the Whisper model used for audio transcription."""
        if self.whisper_model is not None:
            self.output_queue.put(("status", "Whisper model already loaded.", self.theme.colors.success_light))
            self.output_queue.put(("model_ready", "READY"))
            return

        try:
            # Lazy import to speed up GUI startup
            from faster_whisper import WhisperModel, BatchedInferencePipeline

            self.output_queue.put(("status", "Loading Whisper model...", self.theme.colors.warning))
            self.output_queue.put(("progress", f"Loading Whisper '{self._realtime_model_name}' model (~3GB first download)...\n"))

            device, compute_type = get_whisper_device_and_compute_type(verbose=False)
            if device == "cpu":
                self.output_queue.put(("progress", "CUDA backend unavailable; using CPU...\n"))

            try:
                base_model = WhisperModel(self._realtime_model_name, device=device, compute_type=compute_type)
                # Wrap in BatchedInferencePipeline for parallel batch processing (20-40% speedup)
                self.whisper_model = BatchedInferencePipeline(model=base_model)
            except Exception as exc:
                message = str(exc).lower()
                if device == "cuda" and ("cublas" in message or "cudart" in message or "cannot be loaded" in message):
                    # Missing CUDA DLLs - retry on CPU automatically
                    device = "cpu"
                    compute_type = "int8"
                    self.output_queue.put(("progress", "CUDA libraries missing; retrying on CPU...\n"))
                    base_model = WhisperModel(self._realtime_model_name, device=device, compute_type=compute_type)
                    self.whisper_model = BatchedInferencePipeline(model=base_model)
                else:
                    raise

            self._whisper_device = device

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

            self.output_queue.put(("progress", f"Whisper '{self._realtime_model_name}' loaded on {device.upper()}\n"))
            self.output_queue.put(("status", f"Whisper model ready ({device.upper()})", self.theme.colors.success_light))
            self.output_queue.put(("live_preview", f"Whisper model loaded on {device.upper()}! Recording will start now..."))
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
                self.output_queue.put(("live_preview", ""))
                return

            audio_array = np.asarray(audio_buffer, dtype=np.float32)

            # Audio preprocessing (noise reduction + loudness normalization)
            if transcription_config.noise_reduction_enabled or transcription_config.normalize_audio:
                self.output_queue.put(("progress", "Preprocessing audio (noise reduction/normalization)...\n"))
                audio_array = preprocess_array(
                    audio_array,
                    sample_rate=self.config.recording.sample_rate,
                    noise_reduction=transcription_config.noise_reduction_enabled,
                    normalize=transcription_config.normalize_audio,
                )

            self.output_queue.put(("progress", f"\nTranscribing {duration_seconds}s of audio with Whisper...\n"))

            batch_size = transcription_config.batch_size
            device_for_batch = getattr(self, "_whisper_device", "cpu")
            if device_for_batch == "cpu":
                batch_size = min(batch_size, 8)

            # Use batch processing for 20-40% faster transcription on GPU
            segments, _info = self.whisper_model.transcribe(
                audio_array,
                language=transcription_config.language,
                beam_size=transcription_config.beam_size,
                temperature=transcription_config.temperature,
                condition_on_previous_text=transcription_config.condition_on_previous_text,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=transcription_config.no_speech_threshold,
                hallucination_silence_threshold=transcription_config.hallucination_silence_threshold,
                word_timestamps=transcription_config.word_timestamps,
                vad_filter=transcription_config.vad_filter,
                repetition_penalty=transcription_config.repetition_penalty,
                no_repeat_ngram_size=transcription_config.no_repeat_ngram_size,
                batch_size=batch_size,
                initial_prompt=transcription_config.initial_prompt,
                hotwords=transcription_config.hotwords,
            )

            segments_list = list(segments)
            segments_data: TranscriptSegments = [
                make_transcript_segment(start=segment.start, end=segment.end, text=segment.text.strip())
                for segment in segments_list
            ]

            final_transcript, segments_data, removed_count, hallucination_count = _normalize_transcript_segments(
                segments_data,
                clean_fillers=transcription_config.clean_filler_words,
                filter_hallucinated=True,
            )
            if removed_count > 0:
                self.output_queue.put(("progress", f"Removed {removed_count} repetitive segments\n"))

            # Filter out known hallucination phrases (e.g., "Thank you for watching", "Продолжение следует")
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
                self.output_queue.put(("live_preview", ""))

                status_msg = _build_transcription_complete_status(
                    grammar_enhanced=grammar_enhanced,
                    word_count=word_count,
                )
                self.output_queue.put(("status", status_msg, self.theme.colors.success_light))
            else:
                if hallucination_count > 0:
                    self.output_queue.put(("progress", "Only hallucinations detected - no real speech found\n"))
                    self.output_queue.put(("status", "No real speech detected (hallucinations filtered)", self.theme.colors.warning))
                else:
                    self.output_queue.put(("progress", "No speech detected in recording\n"))
                    self.output_queue.put(("status", "No speech detected", self.theme.colors.warning))
                self.output_queue.put(("live_preview", ""))
                self.output_queue.put(("segments", []))

        except RuntimeError as exc:
            message = str(exc).lower()
            if ("cublas" in message or "cudart" in message or "cannot be loaded" in message) and not _retrying:
                self.output_queue.put(("progress", "CUDA libraries missing; retrying transcription on CPU...\n"))
                try:
                    from faster_whisper import WhisperModel, BatchedInferencePipeline
                    base_model = WhisperModel(self._realtime_model_name, device="cpu", compute_type="int8")
                    self.whisper_model = BatchedInferencePipeline(model=base_model)
                    self._whisper_device = "cpu"
                except Exception as reload_exc:
                    self.output_queue.put(("error", f"Transcription error: {reload_exc}"))
                    return
                self._transcribe_accumulated_audio(
                    audio_buffer,
                    duration_seconds,
                    transcription_config,
                    grammar_config,
                    _retrying=True,
                )
            elif "out of memory" in message and self._whisper_device == "cuda":
                self.output_queue.put(("error", "GPU out of memory! Try recording shorter segments."))
            else:
                self.output_queue.put(("error", f"Transcription error: {exc}"))
        except Exception as exc:
            self.output_queue.put(("error", f"Transcription error: {exc}"))
        finally:
            self.output_queue.put(("transcribe_thread_done", ""))

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
            sample_rate = 16000
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
            elif kind == "live_preview":
                self.update_live_preview(msg[1])
            elif kind == "transcript":
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
            elif kind == "transcribe_finished":
                self.transcribe_button.setEnabled(True)
                self.cancel_youtube_button.setVisible(False)
                self.cancel_youtube_button.setEnabled(True)
                # Flash success on transcript card
                if hasattr(self, '_transcript_card'):
                    self._flash_card_success(self._transcript_card)
            elif kind == "transcribe_thread_done":
                self.transcribe_thread = None
            elif kind == "local_file_done":
                self.transcribe_file_button.setEnabled(True)
                self.browse_file_button.setEnabled(True)
                self.transcribe_thread = None
                # Flash success on transcript card
                if hasattr(self, '_transcript_card'):
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
                    self.gpu_status_label.setStyleSheet(self.theme.get_gpu_status_style(True))
                else:
                    self.gpu_status_label.setText("CPU Mode")
                    self.gpu_status_label.setStyleSheet(self.theme.get_gpu_status_style(False))
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
        self.live_preview.clear()
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
                "generated_by": "YouTube Transcriber",
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
    app.setApplicationName("YouTube Transcriber")
    gui = TranscriberGUI()
    gui.run()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "Fatal Error", f"Application error: {exc}")
        sys.exit(1)
