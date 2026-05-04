"""Core transcription functions used by the GUI and CLI.

This module wraps YouTube caption fetching, yt-dlp audio extraction, and local
faster-whisper transcription with optional post-processing.
"""

from __future__ import annotations

import argparse
import array
from collections.abc import Callable, Iterable, Sequence
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import warnings
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Optional, overload

import runtime_bootstrap  # noqa: F401
from app_paths import get_ffmpeg_search_roots, get_log_path, register_windows_dll_directory
from torch_runtime import get_torch

from audio_preprocessor import preprocess_file
from config import TranscriptionConfig, get_config
from exceptions import AudioDownloadError, FileValidationError, TranscriberError
from transcript_types import (
    TranscriptSegments,
    make_transcript_segment,
    replace_segment_text,
)

# YouTube Transcript API import with error handling
try:
    import youtube_transcript_api  # noqa: F401

    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False

# Suppress ctranslate2 pkg_resources deprecation warning (emitted when importing faster-whisper/ctranslate2)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
# Suppress PyTorch CUDA compatibility warning for RTX 5080 (sm_120 Blackwell)
warnings.filterwarnings("ignore", message=".*CUDA capability sm_120.*", category=UserWarning)

# Configure logging
logger = logging.getLogger(__name__)

MEDIA_CMD_TIMEOUT = 30 * 60


def _ctranslate2_cuda_supported(*, verbose: bool = False) -> bool:
    """Check whether faster-whisper's CTranslate2 backend can use CUDA."""
    if sys.platform == "win32" and not ensure_cuda12_runtime_on_windows():
        if verbose:
            logger.warning("CUDA 12 runtime not found (cublas64_12.dll).")
        return False

    try:
        import ctranslate2
    except ImportError:
        if verbose:
            logger.warning("CTranslate2 is not installed; CUDA backend unavailable.")
        return False
    except Exception as exc:
        if verbose:
            logger.warning(f"Failed to import CTranslate2: {exc}")
        return False

    try:
        supported_types = ctranslate2.get_supported_compute_types("cuda")
    except Exception as exc:
        if verbose:
            logger.warning(f"CTranslate2 CUDA probe failed: {exc}")
        return False

    has_cuda = bool(supported_types)
    if verbose:
        if has_cuda:
            logger.info(
                "CTranslate2 CUDA backend available (compute types: %s)",
                ", ".join(sorted(supported_types)),
            )
        else:
            logger.info("CTranslate2 CUDA backend not available; using CPU.")
    return has_cuda


def get_whisper_cuda_status() -> tuple[bool, str]:
    """Return whether faster-whisper can use CUDA and a display name."""
    if not _ctranslate2_cuda_supported(verbose=False):
        return False, ""

    # Prefer concrete device name when PyTorch CUDA is available.
    torch_module = get_torch(context="youtube_transcriber:get_whisper_cuda_status")
    if torch_module is not None:
        try:
            if torch_module.cuda.is_available():
                return True, torch_module.cuda.get_device_name(0)
        except Exception as exc:
            logger.debug(
                "Torch CUDA probe failed while resolving Whisper CUDA status",
                exc_info=exc,
            )

    # Fall back to CTranslate2-level detection if torch is CPU-only.
    try:
        import ctranslate2

        device_count = ctranslate2.get_cuda_device_count()
        if device_count > 0:
            suffix = "device" if device_count == 1 else "devices"
            return True, f"CTranslate2 CUDA ({device_count} {suffix})"
    except Exception as exc:
        logger.debug(
            "CTranslate2 CUDA probe failed while resolving Whisper CUDA status",
            exc_info=exc,
        )

    return True, "CTranslate2 CUDA"


def get_whisper_device_and_compute_type(
    *,
    config: Optional[TranscriptionConfig] = None,
    verbose: bool = True,
) -> tuple[str, str]:
    """Public wrapper used by both CLI and GUI transcription flows."""
    return _setup_device_and_compute_type(config=config, verbose=verbose)


def ensure_cuda12_runtime_on_windows() -> bool:
    """Ensure CUDA 12 runtime DLLs are discoverable for CTranslate2 on Windows.

    CTranslate2 GPU wheels currently depend on CUDA 12.x libraries (notably
    ``cublas64_12.dll``). PyTorch can be built against a different CUDA major
    version (e.g., CUDA 13) and does not satisfy this dependency.

    Returns:
        True when ``cublas64_12.dll`` can be loaded, False otherwise.
    """
    if sys.platform != "win32":
        return True

    try:
        import ctypes
    except ImportError:
        return False

    dll_name = "cublas64_12.dll"
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return False

    def _can_load() -> bool:
        try:
            win_dll(dll_name)
        except OSError:
            return False
        return True

    if _can_load():
        return True

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return False

    candidates: list[Path] = []
    for env_name in ("CUDA_PATH", "CUDA_PATH_V12_0", "CUDA_PATH_V12_1", "CUDA_PATH_V12_2"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value) / "bin")

    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.exists():
        for child in cuda_root.glob("v12.*"):
            candidates.append(child / "bin")

    # If installed via pip, NVIDIA CUDA 12 wheels place DLLs under site-packages/nvidia/*/bin.
    for entry in sys.path:
        try:
            root = Path(entry)
        except TypeError:
            continue
        candidate = root / "nvidia" / "cublas" / "bin"
        if (candidate / dll_name).exists():
            candidates.append(candidate)

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []

    for directory in candidates:
        if not directory.is_dir():
            continue
        directory_str = str(directory)
        if not register_windows_dll_directory(directory_str):
            continue
        if directory_str not in path_entries:
            _prepend_directory_to_path_once(directory_str)
            path_entries.insert(0, directory_str)

        if _can_load():
            logger.info(f"Loaded CUDA runtime from: {directory_str}")
            return True

    return _can_load()


def setup_logging(verbose: bool = False, log_file: str = "youtube_transcriber.log") -> None:
    """Configure logging with console and file handlers.

    Args:
        verbose: Enable debug level logging
        log_file: Path to log file
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')

    # Console handler (simple format for user-facing output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)

    # File handler (detailed format with rotation)
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = get_log_path(log_file)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(detailed_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress verbose output from third-party libraries
    logging.getLogger('yt_dlp').setLevel(logging.WARNING)
    logging.getLogger('faster_whisper').setLevel(logging.INFO)


# Named constants (previously magic numbers)
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
MIN_FREE_DISK_GB = 1.0
PROGRESS_DISPLAY_INTERVAL_S = 2.0
GPU_MEMORY_WARNING_MB = 100
MAX_SPEECH_DURATION_S = 30.0
OPENAI_AUDIO_UPLOAD_LIMIT_BYTES = 25 * 1024 * 1024
OPENAI_AUDIO_SAFE_UPLOAD_LIMIT_BYTES = 24 * 1024 * 1024
OPENAI_DEFAULT_CHUNK_SECONDS = 600.0
OPENAI_MIN_CHUNK_SECONDS = 30.0

# Common video container extensions (audio-only size limit should not apply)
VIDEO_EXTENSIONS: set[str] = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}
OPENAI_SUPPORTED_AUDIO_EXTENSIONS: set[str] = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

# FFmpeg path cache (avoids slow recursive directory walk on each call)
_cached_ffmpeg_path: Optional[str] = None
_ffmpeg_cache_checked = False
_VIDEO_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def _gpu_memory_fraction() -> float:
    return get_config().gpu_memory_fraction


def _normalize_device_preference(config: TranscriptionConfig | None) -> str:
    if config is None:
        return "auto"
    value = str(getattr(config, "device_preference", "auto")).strip().lower()
    if value in {"auto", "cuda", "cpu"}:
        return value
    return "auto"


def _normalize_compute_type_preference(config: TranscriptionConfig | None) -> str:
    if config is None:
        return "auto"
    value = str(getattr(config, "compute_type", "auto")).strip().lower()
    if value in {"auto", "float16", "int8"}:
        return value
    return "auto"


def _cpu_fallback_batch_size(config: TranscriptionConfig) -> int:
    fallback_batch = int(getattr(config, "cpu_fallback_batch_size", 8) or 8)
    return max(1, fallback_batch)


def _max_audio_size_mb() -> int:
    return get_config().max_audio_size_mb


def _max_filename_length() -> int:
    return get_config().max_filename_length


def _log_transcription_runtime_config(config: TranscriptionConfig, *, context: str) -> None:
    """Log the active runtime knobs so GUI progress matches backend behavior."""
    hotword_count = len([term for term in (config.hotwords or "").split(",") if term.strip()])
    logger.info(
        "%s runtime: backend=%s | openai_model=%s | model=%s | device_pref=%s | compute=%s | "
        "beam=%s | batch=%s | cpu_fallback_batch=%s | language=%s | hotwords=%s",
        context,
        getattr(config, "batch_backend", "local_whisper"),
        getattr(config, "openai_batch_model", "gpt-4o-transcribe"),
        config.whisper_model,
        _normalize_device_preference(config),
        _normalize_compute_type_preference(config),
        config.beam_size,
        config.batch_size,
        _cpu_fallback_batch_size(config),
        config.language or "auto",
        f"{hotword_count} hotword(s)" if hotword_count else "none",
    )
    logger.info(
        "%s safeguards: vad=%s | vad_threshold=%.2f | no_speech_threshold=%.2f | "
        "word_timestamps=%s | prev_text=%s | hallucination_filter=%s | dedupe=%s | "
        "filler_cleanup=%s | noise_reduction=%s | normalize_audio=%s",
        context,
        config.vad_filter,
        config.vad_threshold,
        config.no_speech_threshold,
        config.word_timestamps,
        config.condition_on_previous_text,
        config.filter_hallucinations,
        config.deduplicate_repeated_segments,
        config.clean_filler_words,
        config.noise_reduction_enabled,
        config.normalize_audio,
    )


@dataclass(frozen=True)
class _CudaBatchBudget:
    """Estimated GPU batch budget for faster-whisper inference."""

    requested_batch_size: int
    effective_batch_size: int
    target_memory_gb: float | None
    total_memory_gb: float | None
    estimated_model_memory_gb: float
    estimated_per_batch_gb: float


def _estimate_cuda_model_memory_gb(model_name: str, compute_type: str) -> float:
    """Estimate the static GPU footprint for a loaded Whisper model."""
    base_memory_gb = {
        "large-v3": 5.2,
    }.get(model_name, 4.8)

    normalized_compute = str(compute_type).strip().lower()
    if normalized_compute.startswith("int8"):
        return base_memory_gb * 0.7
    if normalized_compute == "float32":
        return base_memory_gb * 1.35
    return base_memory_gb


def _estimate_cuda_per_batch_memory_gb(
    config: TranscriptionConfig,
    *,
    model_name: str,
    compute_type: str,
) -> float:
    """Estimate the incremental GPU memory cost of one decode batch unit."""
    model_scale = _estimate_cuda_model_memory_gb(model_name, compute_type) / 5.2
    beam_factor = 1.0 + (max(config.beam_size - 1, 0) * 0.22)
    patience_factor = 1.0 + (max(config.patience - 1.0, 0.0) * 0.35)
    timestamp_factor = 1.15 if config.word_timestamps else 1.0
    context_factor = 1.05 if config.condition_on_previous_text else 1.0
    return 0.08 * model_scale * beam_factor * patience_factor * timestamp_factor * context_factor


def _detect_cuda_total_memory_gb() -> float | None:
    """Return total CUDA memory in GiB when torch telemetry is available."""
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return None

    try:
        if not torch_module.cuda.is_available():
            return None
        return torch_module.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return None


def _plan_cuda_batch_budget(
    config: TranscriptionConfig,
    *,
    model_name: str,
    compute_type: str,
) -> _CudaBatchBudget:
    """Estimate a safe CUDA batch size for the configured VRAM budget."""
    requested_batch_size = max(1, int(config.batch_size or 1))
    estimated_model_memory_gb = _estimate_cuda_model_memory_gb(model_name, compute_type)
    estimated_per_batch_gb = _estimate_cuda_per_batch_memory_gb(
        config,
        model_name=model_name,
        compute_type=compute_type,
    )
    total_memory_gb = _detect_cuda_total_memory_gb()

    if total_memory_gb is None:
        return _CudaBatchBudget(
            requested_batch_size=requested_batch_size,
            effective_batch_size=requested_batch_size,
            target_memory_gb=None,
            total_memory_gb=None,
            estimated_model_memory_gb=estimated_model_memory_gb,
            estimated_per_batch_gb=estimated_per_batch_gb,
        )

    target_memory_gb = total_memory_gb * _gpu_memory_fraction()
    reserve_memory_gb = max(0.75, total_memory_gb * 0.06)
    available_decode_memory_gb = max(
        target_memory_gb - estimated_model_memory_gb - reserve_memory_gb,
        estimated_per_batch_gb,
    )
    max_batch_size = max(1, int(math.floor(available_decode_memory_gb / estimated_per_batch_gb)))
    effective_batch_size = min(requested_batch_size, max_batch_size)

    return _CudaBatchBudget(
        requested_batch_size=requested_batch_size,
        effective_batch_size=effective_batch_size,
        target_memory_gb=target_memory_gb,
        total_memory_gb=total_memory_gb,
        estimated_model_memory_gb=estimated_model_memory_gb,
        estimated_per_batch_gb=estimated_per_batch_gb,
    )


def _build_runtime_transcribe_kwargs(
    config: TranscriptionConfig,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    context: str,
) -> dict[str, Any]:
    """Build transcribe kwargs and clamp batch size to the active runtime budget."""
    kwargs = _build_transcribe_kwargs(config)
    requested_batch_size = int(kwargs.get("batch_size", config.batch_size) or 1)

    if device == "cpu":
        effective_batch_size = min(requested_batch_size, _cpu_fallback_batch_size(config))
        kwargs["batch_size"] = effective_batch_size
        if effective_batch_size != requested_batch_size:
            logger.info(
                "%s CPU batch budget: requested batch=%s -> effective batch=%s",
                context,
                requested_batch_size,
                effective_batch_size,
            )
        return kwargs

    if device != "cuda":
        return kwargs

    budget = _plan_cuda_batch_budget(
        config,
        model_name=model_name,
        compute_type=compute_type,
    )
    kwargs["batch_size"] = budget.effective_batch_size

    if budget.target_memory_gb is None or budget.total_memory_gb is None:
        logger.info(
            "%s GPU batch budget: no direct VRAM telemetry available; keeping requested batch=%s and using %.0f%% target as best effort.",
            context,
            requested_batch_size,
            _gpu_memory_fraction() * 100,
        )
        return kwargs

    logger.info(
        "%s GPU batch budget: target<=%.1f GB (%.0f%% of %.1f GB) | estimated model=%.1f GB | "
        "per batch=%.2f GB | requested batch=%s -> effective batch=%s",
        context,
        budget.target_memory_gb,
        _gpu_memory_fraction() * 100,
        budget.total_memory_gb,
        budget.estimated_model_memory_gb,
        budget.estimated_per_batch_gb,
        budget.requested_batch_size,
        budget.effective_batch_size,
    )
    return kwargs


def _prepend_directory_to_path_once(directory: str) -> None:
    """Prepend a directory to PATH without duplicating it."""
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    normalized_entries = {os.path.normcase(entry) for entry in path_entries}
    if os.path.normcase(directory) in normalized_entries:
        return
    os.environ["PATH"] = directory + os.pathsep + current_path if current_path else directory


def _truncate_filename(filename: str, *, max_length: int) -> str:
    """Truncate a filename while preserving its extension when practical."""
    if len(filename) <= max_length:
        return filename

    path = Path(filename)
    suffix = "".join(path.suffixes)
    if suffix and len(suffix) < max_length:
        stem = filename[: -len(suffix)]
        truncated_stem = stem[: max_length - len(suffix)].rstrip(" .")
        if truncated_stem:
            return f"{truncated_stem}{suffix}"

    return filename[:max_length].rstrip(" .")


def sanitize_filename(filename: str, max_length: int | None = None) -> str:
    """Sanitize filename to prevent path traversal and invalid characters.

    Args:
        filename: The filename to sanitize
        max_length: Maximum length for the filename

    Returns:
        Sanitized filename safe for the current platform
    """
    if max_length is None:
        max_length = _max_filename_length()

    # Remove any path components (prevents path traversal)
    filename = os.path.basename(filename)

    # Remove traversal attempts
    filename = filename.replace("..", "")

    if sys.platform == "win32":
        # Windows reserved characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)
        filename = filename.strip().rstrip(" .")

        path = Path(filename)
        suffix = "".join(path.suffixes)
        stem = filename[: -len(suffix)] if suffix else filename
        stem = stem.rstrip(" .")
        if stem.upper() in _WINDOWS_RESERVED_BASENAMES:
            stem = f"_{stem}"
        if not stem:
            stem = "untitled_video"
        filename = f"{stem}{suffix}"
    else:
        # Unix-like systems (remove / and null)
        filename = re.sub(r"[/\x00]", "", filename)

    # Ensure filename is not empty after sanitization
    if not filename or filename.strip() == "":
        filename = "untitled_video"

    # Limit length
    return _truncate_filename(filename, max_length=max_length).strip().rstrip(" .") or "untitled_video"


def _normalise_host(netloc: str) -> str:
    """Extract hostname portion without port."""
    return netloc.split(":")[0].lower()


def _coerce_video_id(candidate: str | None) -> Optional[str]:
    """Return a normalized YouTube video ID when the candidate is valid."""
    if candidate is None:
        return None
    stripped = candidate.strip()
    if _VIDEO_ID_RE.fullmatch(stripped):
        return stripped
    return None


def validate_youtube_url(url: str) -> tuple[bool, str | None]:
    """Validate that the URL is a legitimate YouTube URL.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(url, str) or not url.strip():
        return False, "URL is empty"

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as e:
        return False, f"Invalid URL format: {e}"

    if parsed.scheme not in {"http", "https"}:
        return False, "URL must start with http:// or https://"

    host = _normalise_host(parsed.netloc)
    if not host:
        return False, "URL is missing a hostname"

    allowed_hosts = {
        "youtube.com",
        "youtube-nocookie.com",
        "youtu.be",
    }
    if not (
        host in allowed_hosts
        or host.endswith(".youtube.com")
        or host.endswith(".youtube-nocookie.com")
    ):
        return False, f"Invalid domain: {parsed.netloc}. Expected a YouTube URL."

    if extract_video_id(url) is None:
        return False, "No valid video ID found in URL"
    return True, None


def check_dependencies(*, include_gpu_probe: bool = True) -> tuple[bool, list[str]]:
    """Check if all required dependencies are available.

    Returns:
        Tuple of (all_ok, missing_dependencies)
    """
    missing = []

    # Check FFmpeg and ffprobe
    try:
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path:
            logger.debug(f"FFmpeg found at: {ffmpeg_path}")
        else:
            try:
                subprocess.run(
                    ["ffmpeg", "-version"],
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                subprocess.run(
                    ["ffprobe", "-version"],
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                logger.debug("FFmpeg and ffprobe found in system PATH")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                missing.append("FFmpeg/ffprobe (required for audio processing)")
    except Exception as e:
        logger.warning(f"Error checking FFmpeg: {e}")
        missing.append("FFmpeg/ffprobe (check failed)")

    # Check yt-dlp
    try:
        import yt_dlp as _test_yt_dlp  # noqa: F401
        logger.debug("yt-dlp is installed")
    except ImportError:
        missing.append("yt-dlp (required for YouTube downloads)")

    # Check faster-whisper
    try:
        from faster_whisper import WhisperModel as _test_whisper  # noqa: F401
        logger.debug("faster-whisper is installed")
    except ImportError:
        missing.append("faster-whisper (required for transcription)")

    # Check torch (optional for startup; faster-whisper uses CTranslate2)
    torch_module = get_torch(context="youtube_transcriber:check_dependencies")
    if torch_module is not None:
        logger.debug("torch is installed")
    else:
        logger.warning("torch is unavailable; torch-backed features may be limited")

    # Check sounddevice
    try:
        import sounddevice as _test_sd  # noqa: F401
        logger.debug("sounddevice is installed")
    except ImportError:
        missing.append("sounddevice (required for microphone recording)")

    if include_gpu_probe:
        # Check faster-whisper CUDA backend availability (warning, not error).
        whisper_cuda_ok, gpu_name = get_whisper_cuda_status()
        if whisper_cuda_ok:
            if gpu_name:
                logger.info(f"Whisper CUDA backend available: {gpu_name}")
            else:
                logger.info("Whisper CUDA backend available")
        else:
            logger.warning("Whisper CUDA backend not available - will use CPU (slower)")

    all_ok = len(missing) == 0
    return all_ok, missing


def find_ffmpeg() -> Optional[str]:
    """Try to locate FFmpeg installation on Windows.

    Uses caching to avoid slow recursive directory walks on repeated calls.

    Returns:
        Path to FFmpeg directory or None if not found
    """
    global _cached_ffmpeg_path, _ffmpeg_cache_checked

    # Return cached result if we've already searched
    if _ffmpeg_cache_checked:
        return _cached_ffmpeg_path

    # Common installation paths
    common_paths = [str(path) for path in get_ffmpeg_search_roots()] + [
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages"
        ),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "FFmpeg"),
        os.path.join(
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "FFmpeg"
        ),
        "C:\\ffmpeg",
    ]

    def _directory_has_ffmpeg_tools(directory: str) -> bool:
        return all(
            os.path.exists(os.path.join(directory, executable))
            for executable in ("ffmpeg.exe", "ffprobe.exe")
        )

    # Search for ffmpeg.exe and ffprobe.exe in common paths
    for base_path in common_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                if "ffmpeg.exe" in files and "ffprobe.exe" in files:
                    ffmpeg_dir = os.path.join(root)
                    logger.info(f"Found FFmpeg at: {ffmpeg_dir}")
                    _cached_ffmpeg_path = ffmpeg_dir
                    _ffmpeg_cache_checked = True
                    return ffmpeg_dir

    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        if path_entry and _directory_has_ffmpeg_tools(path_entry):
            logger.info(f"Found FFmpeg tools in PATH directory: {path_entry}")
            _cached_ffmpeg_path = path_entry
            _ffmpeg_cache_checked = True
            return path_entry

    _ffmpeg_cache_checked = True
    _cached_ffmpeg_path = None
    return None


def _ffmpeg_executable(name: str, ffmpeg_location: Optional[str]) -> str:
    """Resolve an FFmpeg companion executable (ffmpeg/ffprobe) across platforms."""
    if ffmpeg_location:
        executable_name = f"{name}.exe" if sys.platform == "win32" else name
        candidate = Path(ffmpeg_location) / executable_name
        if candidate.exists():
            return str(candidate)
    return name


def _log_media_cmd_timeout(cmd: Sequence[object], exc: subprocess.TimeoutExpired) -> None:
    timeout = exc.timeout if exc.timeout is not None else MEDIA_CMD_TIMEOUT
    logger.warning("Media command timed out after %ss: %s", timeout, " ".join(str(part) for part in cmd))


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class AudioStreamInfo:
    """Metadata for an audio stream as reported by ffprobe."""

    audio_index: int  # 0-based index within audio streams (maps to ffmpeg -map 0:a:{audio_index})
    codec_name: str | None
    channels: int | None
    sample_rate_hz: int | None
    bit_rate_bps: int | None
    language: str | None
    title: str | None


@dataclass(frozen=True)
class AudioStreamCandidate:
    """Audio stream plus a quick energy estimate for selection."""

    info: AudioStreamInfo
    rms: float
    peak: float
    probes: tuple[tuple[float, float, float], ...]

    @property
    def audio_index(self) -> int:
        return self.info.audio_index

    def describe(self) -> str:
        parts = [f"a:{self.audio_index}"]
        if self.info.codec_name:
            parts.append(self.info.codec_name)
        if self.info.channels:
            parts.append(f"{self.info.channels}ch")
        if self.info.sample_rate_hz:
            parts.append(f"{self.info.sample_rate_hz}Hz")
        if self.info.bit_rate_bps:
            parts.append(f"{self.info.bit_rate_bps//1000}kbps")
        if self.info.language:
            parts.append(self.info.language)
        if self.info.title:
            parts.append(self.info.title)
        parts.append(f"rms={self.rms:.4f}")
        parts.append(f"peak={self.peak:.4f}")
        if self.probes:
            probe_str = ", ".join(
                f"{offset:.0f}s:{rms:.4f}/{peak:.4f}" for offset, rms, peak in self.probes
            )
            parts.append(f"probes={probe_str}")
        return " | ".join(parts)


@dataclass(frozen=True)
class _OpenAIAudioChunk:
    path: str
    start: float
    end: float


def _list_audio_streams(file_path: str, ffmpeg_location: Optional[str]) -> list[AudioStreamInfo]:
    """List audio streams using ffprobe (best-effort; returns empty on failure)."""
    ffprobe = _ffmpeg_executable("ffprobe", ffmpeg_location)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_name,channels,sample_rate,bit_rate:stream_tags=language,title",
        "-of",
        "json",
        file_path,
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=MEDIA_CMD_TIMEOUT)
    except FileNotFoundError:
        logger.debug("ffprobe not found; skipping audio stream enumeration")
        return []
    except subprocess.TimeoutExpired as exc:
        _log_media_cmd_timeout(cmd, exc)
        return []
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        logger.debug(f"ffprobe failed; skipping audio stream enumeration: {stderr}")
        return []

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return []

    streams = payload.get("streams")
    if not isinstance(streams, list):
        return []

    infos: list[AudioStreamInfo] = []
    for audio_index, stream in enumerate(streams):
        if not isinstance(stream, dict):
            continue
        raw_tags = stream.get("tags")
        tags: dict[str, object] = raw_tags if isinstance(raw_tags, dict) else {}
        raw_language = tags.get("language")
        raw_title = tags.get("title")
        infos.append(
            AudioStreamInfo(
                audio_index=audio_index,
                codec_name=stream.get("codec_name") if isinstance(stream.get("codec_name"), str) else None,
                channels=_parse_int(stream.get("channels")),
                sample_rate_hz=_parse_int(stream.get("sample_rate")),
                bit_rate_bps=_parse_int(stream.get("bit_rate")),
                language=raw_language if isinstance(raw_language, str) else None,
                title=raw_title if isinstance(raw_title, str) else None,
            )
        )
    return infos


def _probe_audio_energy(
    file_path: str,
    audio_index: int,
    ffmpeg_location: Optional[str],
    *,
    probe_seconds: int = 15,
    start_seconds: float = 0.0,
) -> tuple[float, float]:
    """Estimate audio RMS/peak from a short ffmpeg decode of the selected stream."""
    ffmpeg = _ffmpeg_executable("ffmpeg", ffmpeg_location)
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if start_seconds > 0:
        # Fast seek is good enough for energy estimation.
        cmd.extend(["-ss", f"{start_seconds:.3f}"])
    cmd.extend(
        [
            "-i",
            file_path,
            "-map",
            f"0:a:{audio_index}",
            "-t",
            str(probe_seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ]
    )
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, timeout=MEDIA_CMD_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        _log_media_cmd_timeout(cmd, exc)
        return 0.0, 0.0
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0.0, 0.0

    raw = completed.stdout
    if not raw:
        return 0.0, 0.0

    samples = array.array("h")
    try:
        samples.frombytes(raw)
    except ValueError:
        return 0.0, 0.0
    if not samples:
        return 0.0, 0.0

    sum_sq = 0.0
    peak = 0
    for sample in samples:
        abs_sample = abs(sample)
        if abs_sample > peak:
            peak = abs_sample
        sum_sq += float(sample) * float(sample)

    rms = math.sqrt(sum_sq / len(samples)) / 32768.0
    peak_norm = peak / 32768.0
    return rms, peak_norm


def _rank_audio_streams_for_transcription(
    file_path: str, ffmpeg_location: Optional[str]
) -> list[AudioStreamCandidate]:
    streams = _list_audio_streams(file_path, ffmpeg_location)
    if not streams:
        return []

    duration = _ffprobe_duration_seconds(file_path, ffmpeg_location)
    probe_seconds = 15
    offsets: list[float] = [0.0]
    if duration and duration > probe_seconds * 4:
        offsets.extend([duration * 0.33, duration * 0.66])
    offsets = sorted({max(0.0, float(offset)) for offset in offsets})

    candidates: list[AudioStreamCandidate] = []
    for stream in streams:
        probes: list[tuple[float, float, float]] = []
        max_rms = 0.0
        max_peak = 0.0
        for offset in offsets:
            rms, peak = _probe_audio_energy(
                file_path,
                stream.audio_index,
                ffmpeg_location,
                probe_seconds=probe_seconds,
                start_seconds=offset,
            )
            probes.append((offset, rms, peak))
            if rms > max_rms:
                max_rms = rms
            if peak > max_peak:
                max_peak = peak
        candidates.append(
            AudioStreamCandidate(
                info=stream,
                rms=max_rms,
                peak=max_peak,
                probes=tuple(probes),
            )
        )
    candidates.sort(key=lambda candidate: (candidate.rms, candidate.peak), reverse=True)
    return candidates


def _extract_audio_to_wav(
    file_path: str,
    output_path: str,
    ffmpeg_location: Optional[str],
    *,
    audio_index: int | None = None,
    duration_seconds: int | None = None,
    gain_db: float | None = None,
) -> None:
    ffmpeg = _ffmpeg_executable("ffmpeg", ffmpeg_location)
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-i",
        file_path,
    ]
    if audio_index is not None:
        cmd.extend(["-map", f"0:a:{audio_index}"])
    if duration_seconds is not None:
        cmd.extend(["-t", str(duration_seconds)])
    if gain_db is not None and gain_db != 0:
        cmd.extend(["-af", f"volume={gain_db}dB"])
    cmd.extend(
        [
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=MEDIA_CMD_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        _log_media_cmd_timeout(cmd, exc)
        raise


def _extract_audio_to_openai_mp3(
    file_path: str,
    output_path: str,
    ffmpeg_location: Optional[str],
    *,
    audio_index: int | None = None,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Extract a mono MP3 that is small enough to upload to the OpenAI Audio API."""
    ffmpeg = _ffmpeg_executable("ffmpeg", ffmpeg_location)
    cmd = [ffmpeg, "-y", "-v", "error", "-nostdin"]
    if start_seconds is not None and start_seconds > 0:
        cmd.extend(["-ss", f"{start_seconds:.3f}"])
    cmd.extend(["-i", file_path])
    if audio_index is not None:
        cmd.extend(["-map", f"0:a:{audio_index}"])
    if duration_seconds is not None:
        cmd.extend(["-t", f"{duration_seconds:.3f}"])
    cmd.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "96k",
            "-codec:a",
            "libmp3lame",
            output_path,
        ]
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=MEDIA_CMD_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        _log_media_cmd_timeout(cmd, exc)
        raise


def _wav_duration_seconds(path: str) -> float | None:
    """Return WAV duration in seconds for PCM WAV files."""
    if not path.lower().endswith(".wav"):
        return None
    try:
        import wave
    except ImportError:
        return None
    try:
        with wave.open(path, "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return wav_file.getnframes() / float(frame_rate)
    except (wave.Error, OSError):
        return None


def _ffprobe_duration_seconds(file_path: str, ffmpeg_location: Optional[str]) -> float | None:
    ffprobe = _ffmpeg_executable("ffprobe", ffmpeg_location)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=MEDIA_CMD_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        _log_media_cmd_timeout(cmd, exc)
        return None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    output = (completed.stdout or "").strip()
    try:
        return float(output)
    except ValueError:
        return None


def _probe_duration_seconds(file_path: str, ffmpeg_location: Optional[str]) -> float | None:
    return _wav_duration_seconds(file_path) or _ffprobe_duration_seconds(file_path, ffmpeg_location)


def is_openai_api_configured() -> bool:
    """Return whether an OpenAI API key is available without exposing the key."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _require_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise TranscriberError("OPENAI_API_KEY is not set. Rotate the exposed key and set a fresh key in the environment.")
    return api_key


def _build_openai_transcription_prompt(config: TranscriptionConfig) -> str | None:
    prompt_parts: list[str] = []
    if config.initial_prompt:
        prompt_parts.append(config.initial_prompt.strip())
    if config.hotwords:
        hotwords = ", ".join(term.strip() for term in config.hotwords.split(",") if term.strip())
        if hotwords:
            prompt_parts.append(f"Vocabulary and names that may appear: {hotwords}.")
    if not prompt_parts:
        return None
    return "\n".join(prompt_parts)


def _extract_openai_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(response, dict):
        raw_text = response.get("text")
        if isinstance(raw_text, str):
            return raw_text.strip()
    return str(response).strip()


def _create_openai_client() -> Any:
    api_key = _require_openai_api_key()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise TranscriberError("The openai package is required for OpenAI transcription. Run pip install -r requirements.txt.") from exc
    return OpenAI(api_key=api_key)


def _transcribe_openai_chunk(client: Any, chunk: _OpenAIAudioChunk, config: TranscriptionConfig) -> str:
    model_name = getattr(config, "openai_batch_model", "gpt-4o-transcribe") or "gpt-4o-transcribe"
    request: dict[str, Any] = {
        "model": model_name,
        "response_format": "text",
    }
    if config.language:
        request["language"] = config.language
    prompt = _build_openai_transcription_prompt(config)
    if prompt:
        request["prompt"] = prompt

    with open(chunk.path, "rb") as audio_file:
        response = client.audio.transcriptions.create(file=audio_file, **request)
    return _extract_openai_response_text(response)


def _prepare_openai_audio_chunks(
    file_path: str,
    *,
    ffmpeg_location: Optional[str],
    temp_paths: list[str],
    audio_index: int | None = None,
    force_extract: bool = False,
) -> list[_OpenAIAudioChunk]:
    """Prepare one or more Audio API upload files from local media."""
    file_ext = os.path.splitext(file_path)[1].lower()
    duration = _probe_duration_seconds(file_path, ffmpeg_location)
    file_size = os.path.getsize(file_path)

    if (
        not force_extract
        and file_ext in OPENAI_SUPPORTED_AUDIO_EXTENSIONS
        and file_size <= OPENAI_AUDIO_SAFE_UPLOAD_LIMIT_BYTES
    ):
        end = duration if duration is not None else 0.0
        return [_OpenAIAudioChunk(path=file_path, start=0.0, end=end)]

    import tempfile

    if duration is None:
        fd, extracted_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        temp_paths.append(extracted_path)
        _extract_audio_to_openai_mp3(
            file_path,
            extracted_path,
            ffmpeg_location,
            audio_index=audio_index,
        )
        extracted_size = os.path.getsize(extracted_path)
        if extracted_size > OPENAI_AUDIO_UPLOAD_LIMIT_BYTES:
            raise TranscriberError(
                "OpenAI transcription input is larger than the Audio API upload limit and duration could not be probed for chunking."
            )
        return [_OpenAIAudioChunk(path=extracted_path, start=0.0, end=0.0)]

    chunks: list[_OpenAIAudioChunk] = []
    start = 0.0
    chunk_seconds = min(OPENAI_DEFAULT_CHUNK_SECONDS, max(duration, OPENAI_MIN_CHUNK_SECONDS))
    while start < duration:
        current_duration = min(chunk_seconds, duration - start)
        fd, chunk_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            _extract_audio_to_openai_mp3(
                file_path,
                chunk_path,
                ffmpeg_location,
                audio_index=audio_index,
                start_seconds=start,
                duration_seconds=current_duration,
            )
            chunk_size = os.path.getsize(chunk_path)
            if chunk_size > OPENAI_AUDIO_SAFE_UPLOAD_LIMIT_BYTES and chunk_seconds > OPENAI_MIN_CHUNK_SECONDS:
                os.unlink(chunk_path)
                chunk_seconds = max(OPENAI_MIN_CHUNK_SECONDS, chunk_seconds / 2)
                continue
            if chunk_size > OPENAI_AUDIO_UPLOAD_LIMIT_BYTES:
                os.unlink(chunk_path)
                raise TranscriberError(
                    "OpenAI transcription chunk exceeds the Audio API upload limit even at the minimum chunk size."
                )
            temp_paths.append(chunk_path)
            chunks.append(
                _OpenAIAudioChunk(
                    path=chunk_path,
                    start=start,
                    end=min(start + current_duration, duration),
                )
            )
            start += current_duration
        except Exception:
            if os.path.exists(chunk_path) and chunk_path not in temp_paths:
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass
            raise

    return chunks


def _transcribe_openai_chunks(
    chunks: list[_OpenAIAudioChunk],
    *,
    config: TranscriptionConfig,
    context: str,
) -> tuple[str, TranscriptSegments]:
    client = _create_openai_client()
    segments: TranscriptSegments = []
    logger.info("%s OpenAI model: %s", context, getattr(config, "openai_batch_model", "gpt-4o-transcribe"))
    logger.info("%s OpenAI chunks: %d", context, len(chunks))
    for index, chunk in enumerate(chunks, start=1):
        logger.info(
            "%s OpenAI chunk %d/%d: %.1fs -> %.1fs",
            context,
            index,
            len(chunks),
            chunk.start,
            chunk.end,
        )
        text = _transcribe_openai_chunk(client, chunk, config)
        if text:
            segments.append(make_transcript_segment(start=chunk.start, end=chunk.end, text=text))

    full_transcript, segments, removed_count, hallucination_count = _normalize_transcript_segments(
        segments,
        clean_fillers=config.clean_filler_words,
        filter_hallucinated=config.filter_hallucinations,
        deduplicate_repetitions=config.deduplicate_repeated_segments,
    )
    if removed_count > 0:
        logger.info("Removed %d repetitive OpenAI segment(s) from transcript", removed_count)
    if hallucination_count > 0:
        logger.info("Filtered %d hallucination segment(s) from OpenAI transcript", hallucination_count)
    return full_transcript, segments


def transcribe_audio_openai(
    audio_file: str,
    ffmpeg_location: Optional[str] = None,
    config: Optional[TranscriptionConfig] = None,
    cleanup_audio_file: bool = False,
) -> tuple[str | None, TranscriptSegments | None]:
    """Transcribe an audio file with OpenAI's Audio API."""
    if config is None:
        config = get_config().transcription
    temp_paths: list[str] = []
    try:
        _log_transcription_runtime_config(config, context="OpenAI/audio transcription")
        chunks = _prepare_openai_audio_chunks(
            audio_file,
            ffmpeg_location=ffmpeg_location,
            temp_paths=temp_paths,
        )
        return _transcribe_openai_chunks(chunks, config=config, context="OpenAI/audio transcription")
    except Exception as exc:
        logger.exception("Error transcribing audio with OpenAI: %s", exc)
        return None, None
    finally:
        cleanup_paths: list[str] = []
        for candidate in temp_paths:
            if candidate not in cleanup_paths:
                cleanup_paths.append(candidate)
        if cleanup_audio_file and audio_file and audio_file not in cleanup_paths:
            cleanup_paths.append(audio_file)
        for temp_path in cleanup_paths:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.info(f"Cleaned up temporary file: {temp_path}")
            except OSError:
                pass


def transcribe_local_file_openai(
    file_path: str,
    ffmpeg_location: Optional[str] = None,
    config: Optional[TranscriptionConfig] = None,
) -> tuple[str, TranscriptSegments]:
    """Transcribe a local audio/video file with OpenAI's Audio API."""
    if config is None:
        config = get_config().transcription
    if not os.path.exists(file_path):
        raise FileValidationError(f"File not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()
    file_size_mb = os.path.getsize(file_path) / (1024**2)
    logger.info("Processing local file with OpenAI: %s (%.1fMB)", file_path, file_size_mb)
    _log_transcription_runtime_config(config, context="OpenAI local-file transcription")

    temp_paths: list[str] = []
    try:
        force_extract = file_ext in VIDEO_EXTENSIONS or file_ext not in OPENAI_SUPPORTED_AUDIO_EXTENSIONS
        selected_audio_index: int | None = None
        if file_ext in VIDEO_EXTENSIONS:
            ranked_audio_streams = _rank_audio_streams_for_transcription(file_path, ffmpeg_location)
            if ranked_audio_streams:
                selected_audio_index = ranked_audio_streams[0].audio_index
                logger.info("OpenAI selected audio stream: %s", ranked_audio_streams[0].describe())
            else:
                selected_audio_index = 0
                logger.info("Could not enumerate audio streams for OpenAI; using default audio track (a:0).")

        chunks = _prepare_openai_audio_chunks(
            file_path,
            ffmpeg_location=ffmpeg_location,
            temp_paths=temp_paths,
            audio_index=selected_audio_index,
            force_extract=force_extract,
        )
        return _transcribe_openai_chunks(chunks, config=config, context="OpenAI local-file transcription")
    except (FileValidationError, TranscriberError):
        raise
    except Exception as exc:
        logger.exception("Error transcribing local file with OpenAI")
        raise TranscriberError(f"Error transcribing local file with OpenAI: {exc}") from exc
    finally:
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError:
                pass


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL.

    Args:
        url: YouTube URL or video ID

    Returns:
        Video ID string or None if not found
    """
    stripped = url.strip()
    direct_video_id = _coerce_video_id(stripped)
    if direct_video_id is not None:
        return direct_video_id

    try:
        parsed = urllib.parse.urlparse(stripped)
    except Exception:
        return None

    host = _normalise_host(parsed.netloc)
    path_parts = [part for part in parsed.path.split("/") if part]

    if host == "youtu.be":
        if not path_parts:
            return None
        return _coerce_video_id(path_parts[0])

    if host in {"youtube.com", "youtube-nocookie.com"} or host.endswith(".youtube.com") or host.endswith(".youtube-nocookie.com"):
        query = urllib.parse.parse_qs(parsed.query)
        for key in ("v", "vi"):
            candidate_values = query.get(key)
            if candidate_values:
                video_id = _coerce_video_id(candidate_values[0])
                if video_id is not None:
                    return video_id

        if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live", "v"}:
            return _coerce_video_id(path_parts[1])

    return None


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS timestamp format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript_with_timestamps(segments_data: TranscriptSegments) -> str:
    """Format transcript with timestamps for each segment.

    Args:
        segments_data: List of dicts with 'start', 'end', 'text', and optionally 'speaker'

    Returns:
        Formatted transcript with timestamps
    """
    formatted_lines = []
    for item in segments_data:
        start = item.get('start', 0)
        text = item.get('text', '')
        speaker = item.get('speaker', None)
        timestamp = format_timestamp(start)
        if speaker:
            formatted_lines.append(f"[{timestamp}] {speaker}: {text.strip()}")
        else:
            formatted_lines.append(f"[{timestamp}] {text.strip()}")
    return "\n".join(formatted_lines)


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_transcript_as_srt(segments_data: TranscriptSegments) -> str:
    """Format transcript segments as SRT subtitle file.

    Args:
        segments_data: List of dicts with 'start', 'end', 'text', and optionally 'speaker'

    Returns:
        SRT formatted string
    """
    srt_lines = []
    for idx, item in enumerate(segments_data, start=1):
        start = item.get('start', 0)
        end = item.get('end', 0)
        text = item.get('text', '').strip()
        speaker = item.get('speaker', None)

        # Add speaker label to text if available
        if speaker:
            text = f"{speaker}: {text}"

        start_ts = format_srt_timestamp(start)
        end_ts = format_srt_timestamp(end)

        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between subtitles

    return "\n".join(srt_lines)


def format_transcript_as_json(segments_data: TranscriptSegments) -> str:
    """Format transcript segments as JSON.

    Args:
        segments_data: List of dicts with 'start', 'end', 'text', and optionally 'speaker'

    Returns:
        JSON formatted string
    """
    json_data = []
    for item in segments_data:
        segment = {
            'start': item.get('start', 0),
            'end': item.get('end', 0),
            'text': item.get('text', '').strip()
        }
        if 'speaker' in item and item['speaker']:
            segment['speaker'] = item['speaker']
        json_data.append(segment)

    return json.dumps(json_data, indent=2, ensure_ascii=False)


# Known Whisper hallucination phrases (appear when model encounters silence/noise/music)
HALLUCINATION_PHRASES = frozenset({
    # Russian (very common hallucination)
    "продолжение следует",
    "до свидания",
    "спасибо за просмотр",
    # English
    "thank you for watching",
    "thanks for watching",
    "thank you for reading",
    "thank you for listening",
    "thanks for listening",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "see you next time",
    "see you in the next video",
    "bye bye",
    # German
    "vielen dank fürs zuschauen",
    "bis zum nächsten mal",
    # Spanish
    "gracias por ver",
    "hasta la próxima",
    # French
    "merci d'avoir regardé",
    "à bientôt",
    # Chinese
    "谢谢观看",
    "请订阅",
    # Japanese
    "ご視聴ありがとうございました",
    # Korean
    "시청해 주셔서 감사합니다",
    # Portuguese
    "obrigado por assistir",
})


_PROMPT_LEAKAGE_PATTERNS = (
    re.compile(
        r"\bthis is a faithful transcript of spoken audio(?:, including meetings with diverse accents)?\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpreserve (?:the speaker's )?exact wording, filler words, repetitions, "
        r"(?:and )?(?:unfinished thoughts|uncertain phrasing)\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdo not paraphrase(?:, summarize,)?(?: or)? rewrite for grammar\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\buse punctuation and capitalization only to "
        r"(?:reflect the spoken words|make the spoken words readable)\.?",
        re.IGNORECASE,
    ),
)
_PROMPT_LEAKAGE_MARKERS = (
    "faithful transcript of spoken audio",
    "do not paraphrase",
    "rewrite for grammar",
    "use punctuation and capitalization only",
    "spoken words readable",
)


def _strip_transcription_artifacts(text: str) -> tuple[str, bool]:
    """Remove prompt/instruction leakage that should never appear in the transcript."""
    if not text:
        return "", False

    cleaned = text
    removed = False
    for pattern in _PROMPT_LEAKAGE_PATTERNS:
        cleaned, replacements = pattern.subn(" ", cleaned)
        if replacements:
            removed = True

    if not removed:
        return text.strip(), False

    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[{])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]}])", r"\1", cleaned)
    cleaned = cleaned.strip(" \t\r\n-,;:")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, True


def is_hallucination(text: str) -> bool:
    """Check if text is a known Whisper hallucination phrase.

    Args:
        text: The transcribed text to check

    Returns:
        True if the text appears to be a hallucination
    """
    if not text:
        return False

    stripped_text, artifact_removed = _strip_transcription_artifacts(text)
    cleaned = stripped_text.strip().lower()

    # Empty text after cleaning
    if not cleaned:
        return artifact_removed or bool(text.strip())

    if any(marker in cleaned for marker in _PROMPT_LEAKAGE_MARKERS):
        return True

    # Check for known hallucination phrases
    for phrase in HALLUCINATION_PHRASES:
        # Match if the text is mostly just the hallucination phrase
        if phrase in cleaned and len(cleaned) < len(phrase) + 15:
            return True

    return False


def transcription_result_looks_suspicious(
    segments: TranscriptSegments,
    *,
    input_duration: float | None = None,
) -> bool:
    """Detect obviously bad decode output before we trust it as speech."""
    if not segments:
        return True

    texts: list[str] = []
    prompt_leakage_hits = 0
    for seg in segments:
        raw_text = seg.get("text", "")
        if not isinstance(raw_text, str):
            continue
        cleaned_text, artifact_removed = _strip_transcription_artifacts(raw_text)
        if artifact_removed:
            prompt_leakage_hits += 1
        if cleaned_text:
            texts.append(cleaned_text)

    if prompt_leakage_hits > 0:
        return True
    if not texts:
        return True

    normalized_texts = [text.lower() for text in texts]
    unique_texts = set(normalized_texts)

    if len(texts) >= 4 and len(unique_texts) == 1:
        only = next(iter(unique_texts))
        if len(only) <= 80:
            return True

    non_hallucinated = [text for text in texts if not is_hallucination(text)]
    if not non_hallucinated:
        return True

    if input_duration is not None and input_duration >= 180:
        candidate_text = " ".join(non_hallucinated or texts)
        if len(candidate_text.split()) < 15:
            return True

    return False


def build_suspicion_retry_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a safer retry configuration after suspicious output or silence loops."""
    retry_kwargs = dict(kwargs)
    retry_kwargs["initial_prompt"] = None
    retry_kwargs["condition_on_previous_text"] = False
    retry_kwargs["no_speech_threshold"] = max(float(retry_kwargs.get("no_speech_threshold", 0.3)), 0.5)
    retry_kwargs["hallucination_silence_threshold"] = max(
        float(retry_kwargs.get("hallucination_silence_threshold", 0.5)),
        1.0,
    )
    return retry_kwargs


def filter_hallucinations(segments: list[str]) -> tuple[list[str], int]:
    """Filter out hallucination segments from transcript.

    Args:
        segments: List of segment text strings

    Returns:
        Tuple of (filtered_segments, removed_count)
    """
    filtered = []
    removed = 0

    for seg in segments:
        if is_hallucination(seg):
            removed += 1
            logger.warning(f"Filtered hallucination: '{seg.strip()}'")
        else:
            filtered.append(seg)

    if removed > 0:
        logger.info(f"Removed {removed} hallucination segment(s)")

    return filtered, removed


def clean_filler_words(text: str) -> str:
    """Remove common filler words and hesitation sounds from transcribed text.

    Strips sounds like 'umm', 'uh', hedging phrases like 'you know', 'I mean',
    and filler uses of 'basically', 'actually', 'like'.

    Args:
        text: Transcribed text string.

    Returns:
        Cleaned text with fillers removed, or empty string if input was empty.
    """
    if not text:
        return ""

    # Standalone filler sounds (whole word match, case-insensitive)
    filler_sounds = r'\b(?:um+|uh+|er+|ah+|eh+|hm+|hmm+|mm+)\b'
    text = re.sub(filler_sounds, '', text, flags=re.IGNORECASE)

    # Filler phrases (only when followed by comma or at sentence boundaries)
    filler_phrases = [
        r',?\s*\byou know\b\s*,?',
        r',?\s*\bI mean\b\s*,?',
        r',?\s*\bsort of\b\s*,?',
        r',?\s*\bkind of\b\s*,?',
        r'\bbasically\s*,\s*',
        r'\bactually\s*,\s*',
        r'\blike\s*,\s*',
    ]
    for pattern in filler_phrases:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    # Collapse multiple spaces and strip
    text = re.sub(r'\s{2,}', ' ', text).strip()
    # Fix leftover comma-space issues
    text = re.sub(r'\s*,\s*,', ',', text)
    text = re.sub(r'^\s*,\s*', '', text)

    return text


@overload
def deduplicate_segments(
    segments: list[str],
    repetition_threshold: int = 3,
    *,
    return_indices: Literal[False] = False,
) -> tuple[list[str], int]: ...


@overload
def deduplicate_segments(
    segments: list[str],
    repetition_threshold: int = 3,
    *,
    return_indices: Literal[True],
) -> tuple[list[str], int, list[int]]: ...


def deduplicate_segments(
    segments: list[str],
    repetition_threshold: int = 3,
    *,
    return_indices: bool = False,
) -> tuple[list[str], int] | tuple[list[str], int, list[int]]:
    """Remove consecutive repetitive segments from a transcript.

    This prevents infinite loops where Whisper gets stuck repeating the same text.

    Args:
        segments: List of segment text strings
        repetition_threshold: Number of consecutive repeats before deduplication (default: 3)
        return_indices: When True, also return the original indices that were kept

    Returns:
        Tuple of (deduplicated_segments, removed_count[, kept_indices])
    """
    if not segments:
        return (segments, 0, []) if return_indices else (segments, 0)

    original_count = len(segments)
    deduplicated: list[str] = []
    kept_indices: list[int] = []

    i = 0
    while i < len(segments):
        current_segment = segments[i].strip()

        # Count consecutive repetitions
        repeat_count = 1
        while (
            i + repeat_count < len(segments)
            and segments[i + repeat_count].strip() == current_segment
        ):
            repeat_count += 1

        # If segment repeats excessively, keep only first occurrence
        if repeat_count >= repetition_threshold:
            logger.warning(
                f"Detected repetitive segment (repeated {repeat_count}x): "
                f"'{current_segment[:50]}...' - keeping only first occurrence"
            )
            deduplicated.append(segments[i])
            kept_indices.append(i)
            i += repeat_count  # Skip all duplicates
        else:
            # Add all non-excessive repetitions
            for j in range(repeat_count):
                deduplicated.append(segments[i + j])
                kept_indices.append(i + j)
            i += repeat_count

    removed_count = original_count - len(deduplicated)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} repetitive segments from transcript")

    if return_indices:
        return deduplicated, removed_count, kept_indices
    return deduplicated, removed_count


def _normalize_transcript_segments(
    segments_data: TranscriptSegments,
    *,
    clean_fillers: bool,
    filter_hallucinated: bool,
    deduplicate_repetitions: bool,
) -> tuple[str, TranscriptSegments, int, int]:
    """Normalize, filter, and deduplicate transcript segments."""
    normalized_segments = list(segments_data)
    hallucination_count = 0
    artifact_cleanup_count = 0

    if normalized_segments:
        cleaned_segments: TranscriptSegments = []
        for segment in normalized_segments:
            text = segment.get("text", "")
            cleaned_text, artifact_removed = _strip_transcription_artifacts(text)
            if artifact_removed:
                artifact_cleanup_count += 1
            if not cleaned_text:
                if text and text.strip():
                    hallucination_count += 1
                continue
            cleaned_segments.append(replace_segment_text(segment, cleaned_text))
        normalized_segments = cleaned_segments
        if artifact_cleanup_count:
            logger.warning(
                "Removed prompt/instruction leakage from %d segment(s)",
                artifact_cleanup_count,
            )

    if filter_hallucinated and normalized_segments:
        kept_segments: TranscriptSegments = []
        removed_samples: list[str] = []
        for segment in normalized_segments:
            text = segment.get("text", "")
            if is_hallucination(text):
                hallucination_count += 1
                if len(removed_samples) < 3:
                    removed_samples.append(text.strip())
                continue
            kept_segments.append(segment)
        if hallucination_count:
            logger.warning(
                "Filtered %d hallucination segment(s)%s",
                hallucination_count,
                f" (e.g., {removed_samples!r})" if removed_samples else "",
            )
        normalized_segments = kept_segments

    if clean_fillers and normalized_segments:
        normalized_segments = [
            replace_segment_text(segment, clean_filler_words(segment.get("text", "")))
            for segment in normalized_segments
        ]
        logger.info("Filler words cleaned from transcript")

    segment_texts = [segment.get("text", "") for segment in normalized_segments]
    if deduplicate_repetitions and normalized_segments:
        deduplicated_texts, removed_count, kept_indices = deduplicate_segments(
            segment_texts,
            return_indices=True,
        )
        kept_segments = [normalized_segments[index] for index in kept_indices]
    else:
        deduplicated_texts = segment_texts
        removed_count = 0
        kept_segments = normalized_segments

    normalized_segments = [
        replace_segment_text(segment, deduplicated_text)
        for segment, deduplicated_text in zip(kept_segments, deduplicated_texts, strict=True)
    ]
    return " ".join(deduplicated_texts), normalized_segments, removed_count, hallucination_count


def get_youtube_transcript(video_id: str) -> tuple[str | None, TranscriptSegments | None]:
    """Try to get transcript from YouTube's built-in captions.

    Args:
        video_id: YouTube video ID

    Returns:
        Tuple of (transcript_text, segments_data) where segments_data is a list of dicts
        with 'start', 'end', 'text' keys. Returns (None, None) if unavailable.
    """
    if not TRANSCRIPT_API_AVAILABLE:
        logger.info("YouTube Transcript API not available. Skipping caption check.")
        return None, None

    # Import is guaranteed to be available here due to the check above
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        logger.info(f"Attempting to fetch transcript for video ID: {video_id}")

        # Use new instance-based API (v1.0.0+)
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id).to_raw_data()

        # Combine all transcript segments
        full_transcript = " ".join([item["text"] for item in transcript_data])

        # Extract normalized segment data with timestamps.
        segments_data = [
            make_transcript_segment(
                start=item["start"],
                end=item["start"] + item["duration"],
                text=item["text"],
            )
            for item in transcript_data
        ]

        logger.info("Successfully retrieved YouTube transcript!")
        return full_transcript, segments_data

    except Exception as e:
        logger.warning(f"Could not fetch transcript: {e}")
        logger.info("Will try downloading and transcribing audio instead.")
        return None, None


def download_audio(
    video_url: str, output_path: str = "audio", ffmpeg_location: Optional[str] = None
) -> Optional[str]:
    """Download audio from YouTube video with size limits and error handling.

    Args:
        video_url: YouTube video URL
        output_path: Output path for audio file (without extension)
        ffmpeg_location: Path to FFmpeg directory

    Returns:
        Path to downloaded audio file or None on error.
        This helper intentionally preserves its legacy best-effort contract for
        YouTube fallback callers instead of raising on every failure.

    Raises:
        RuntimeError: If disk space is insufficient or file is too large
    """
    try:
        try:
            import yt_dlp
        except ImportError as exc:
            raise AudioDownloadError("yt-dlp is required for YouTube downloads") from exc

        DownloadError = yt_dlp.utils.DownloadError  # type: ignore[attr-defined]

        logger.info("Downloading audio from YouTube video...")

        # Check available disk space
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        if free_gb < MIN_FREE_DISK_GB:
            raise AudioDownloadError(
                f"Insufficient disk space: {free_gb:.2f}GB free (need at least {MIN_FREE_DISK_GB}GB)"
            )

        # Try to find FFmpeg if not provided
        if not ffmpeg_location:
            logger.info("Searching for FFmpeg installation...")
            ffmpeg_location = find_ffmpeg()

        # Enhanced yt-dlp options with better format fallbacks and error handling
        ydl_opts = {
            # Format selection with fallbacks for signature issues
            # Try multiple format combinations to work around YouTube's SABR streaming
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": output_path,
            "quiet": False,
            "no_warnings": False,  # Show warnings to help diagnose issues
            "socket_timeout": 30,
            "retries": 5,  # Increased from 3 to 5
            "fragment_retries": 5,  # Increased from 3 to 5
            "extractor_retries": 3,  # Retry extractor operations
            # Additional options to work around signature extraction issues
            "nocheckcertificate": False,  # Verify SSL certificates
            "prefer_insecure": False,
            "age_limit": None,
            # Enhanced error handling
            "ignoreerrors": False,  # Don't ignore errors, we want to catch them
            "no_color": True,  # Disable ANSI color codes in output
            # Prevent issues with geo-restricted or age-restricted content
            "geo_bypass": True,
            # Use newer extractors
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"],  # Try multiple clients
                    "player_skip": ["webpage"],  # Skip problematic methods
                }
            },
        }

        # Add FFmpeg location if found
        if ffmpeg_location:
            ydl_opts["ffmpeg_location"] = ffmpeg_location
            logger.info(f"Using FFmpeg from: {ffmpeg_location}")
        else:
            logger.warning("FFmpeg not found in common locations - will try system PATH")

        # Attempt download with enhanced error reporting
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
                logger.info("Starting download with yt-dlp...")
                result = ydl.download([video_url])

                if result != 0:
                    raise AudioDownloadError(
                        f"yt-dlp download failed with exit code {result}. "
                        "This may be due to YouTube signature extraction issues. "
                        "Try updating yt-dlp: pip install --upgrade yt-dlp"
                    )

        except DownloadError as e:
            # Specific handling for yt-dlp download errors
            error_msg = str(e)
            if "Signature extraction failed" in error_msg:
                raise AudioDownloadError(
                    "YouTube signature extraction failed. This usually means:\n"
                    "  1. yt-dlp needs updating: pip install --upgrade yt-dlp\n"
                    "  2. YouTube changed their API (check yt-dlp GitHub for updates)\n"
                    "  3. The video has restrictions (age/geo/private)\n"
                    f"Original error: {error_msg}"
                ) from e
            elif "Video unavailable" in error_msg or "Private video" in error_msg:
                raise AudioDownloadError(
                    f"Video is unavailable or private: {error_msg}"
                ) from e
            else:
                raise AudioDownloadError(f"Download failed: {error_msg}") from e

        audio_file = f"{output_path}.mp3"

        # Validate file exists and check size
        if os.path.exists(audio_file):
            size_mb = os.path.getsize(audio_file) / (1024**2)
            max_audio_size_mb = _max_audio_size_mb()
            if size_mb > max_audio_size_mb:
                os.remove(audio_file)
                raise FileValidationError(
                    f"Audio file too large: {size_mb:.1f}MB (max: {max_audio_size_mb}MB)"
                )
            logger.info(f"Audio downloaded successfully: {audio_file} ({size_mb:.1f}MB)")
        else:
            # Check if file exists with different extension
            possible_files = glob.glob(f"{output_path}.*")
            if possible_files:
                actual_file = possible_files[0]
                logger.warning(f"Expected {audio_file} but found {actual_file}")
                raise AudioDownloadError(
                    f"Audio file conversion may have failed. Found: {actual_file}"
                )
            else:
                raise AudioDownloadError(
                    f"Audio file not found after download: {audio_file}\n"
                    "This may indicate FFmpeg is not installed or not working correctly."
                )

        return audio_file

    except (AudioDownloadError, FileValidationError) as e:
        # Log our custom audio/file errors
        logger.error(str(e))
        return None
    except Exception as e:
        logger.exception("Unexpected error downloading audio: %s", e)
        return None
    finally:
        # Cleanup any partial downloads
        for pattern in [f"{output_path}.*part*", f"{output_path}.*.part"]:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    logger.info(f"Cleaned up partial download: {file}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not remove partial download {file}: {cleanup_error}")


def _build_vad_parameters(config: TranscriptionConfig) -> dict[str, Any]:
    """Build VAD parameters dict from config."""
    return {
        "threshold": config.vad_threshold,
        "min_speech_duration_ms": config.min_speech_duration_ms,
        "max_speech_duration_s": 30.0,
        "min_silence_duration_ms": config.min_silence_duration_ms,
        "speech_pad_ms": config.speech_pad_ms,
    }


def _build_transcribe_kwargs(config: TranscriptionConfig) -> dict[str, Any]:
    """Build the common transcription keyword arguments from config."""
    return {
        "language": config.language,
        "task": "transcribe",
        "beam_size": config.beam_size,
        "best_of": config.beam_size,
        "patience": config.patience,
        "length_penalty": config.length_penalty,
        "temperature": config.temperature,
        "condition_on_previous_text": config.condition_on_previous_text,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": config.no_speech_threshold,
        "hallucination_silence_threshold": config.hallucination_silence_threshold,
        "word_timestamps": config.word_timestamps,
        "vad_filter": config.vad_filter,
        "vad_parameters": _build_vad_parameters(config),
        "batch_size": config.batch_size,
        "repetition_penalty": config.repetition_penalty,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "initial_prompt": config.initial_prompt,
        "hotwords": config.hotwords,
    }


def _maybe_preprocess_audio_path(
    input_path: str,
    *,
    ffmpeg_location: Optional[str],
    config: TranscriptionConfig,
    temp_paths: list[str] | None = None,
) -> str:
    """Apply optional file preprocessing and return the path Whisper should read."""
    if not (config.noise_reduction_enabled or config.normalize_audio):
        return input_path

    import tempfile as _tempfile

    temp_fd, temp_path = _tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)

    ffmpeg_cmd = _ffmpeg_executable("ffmpeg", ffmpeg_location)
    preprocess_input_path = input_path
    decoded_temp_path: str | None = None
    if Path(input_path).suffix.lower() != ".wav":
        decoded_fd, decoded_temp_path = _tempfile.mkstemp(suffix=".wav")
        os.close(decoded_fd)
        decode_cmd = [
            ffmpeg_cmd,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            decoded_temp_path,
        ]
        try:
            subprocess.run(decode_cmd, check=True, capture_output=True, timeout=MEDIA_CMD_TIMEOUT)
            preprocess_input_path = decoded_temp_path
            logger.info("Decoded non-WAV input to temporary WAV before preprocessing")
        except subprocess.TimeoutExpired as exc:
            _log_media_cmd_timeout(decode_cmd, exc)
            logger.warning("Audio decode for preprocessing failed (%s); skipping", exc)
            for candidate in (decoded_temp_path, temp_path):
                if candidate is None:
                    continue
                try:
                    os.unlink(candidate)
                except OSError:
                    pass
            return input_path
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            logger.warning("Audio decode for preprocessing failed (%s); skipping", exc)
            for candidate in (decoded_temp_path, temp_path):
                if candidate is None:
                    continue
                try:
                    os.unlink(candidate)
                except OSError:
                    pass
            return input_path

    result_path: str | None = None
    try:
        result_path = preprocess_file(
            preprocess_input_path,
            temp_path,
            noise_reduction=config.noise_reduction_enabled,
            normalize=config.normalize_audio,
            ffmpeg_cmd=ffmpeg_cmd,
        )
        if result_path != preprocess_input_path:
            if temp_paths is not None:
                temp_paths.append(temp_path)
            logger.info("Audio preprocessing applied")
            return result_path
        return input_path
    finally:
        for candidate in (decoded_temp_path, temp_path):
            if candidate is None or candidate == result_path:
                continue
            try:
                os.unlink(candidate)
            except OSError:
                pass


def _needs_cuda_cpu_fallback(exc: Exception) -> bool:
    """Detect CUDA loader errors that should fall back to CPU."""
    message = str(exc).lower()
    keywords = (
        "cublas",
        "cudart",
        "cudnn",
        "cuda driver",
        "out of memory",
        "no kernel image",
        "unsupported gpu architecture",
        "sm_120",
    )
    return any(keyword in message for keyword in keywords) or ("cannot be loaded" in message and "cuda" in message)


def _is_cuda_out_of_memory_error(exc: BaseException) -> bool:
    """Return True when the runtime failure is specifically a CUDA OOM."""
    return "out of memory" in str(exc).lower()


def _setup_device_and_compute_type(
    *,
    config: Optional[TranscriptionConfig] = None,
    verbose: bool = True,
) -> tuple[str, str]:
    """Determine device and compute type with GPU fallback.

    Checks CUDA availability, CUDA 12 runtime on Windows, and configures
    GPU memory allocation. Falls back to CPU with INT8 if any step fails.

    Args:
        verbose: If True, log detailed GPU information.

    Returns:
        Tuple of (device, compute_type) where device is "cuda" or "cpu"
        and compute_type is "float16" (GPU) or "int8" (CPU).
    """
    device_preference = _normalize_device_preference(config)
    compute_preference = _normalize_compute_type_preference(config)

    if device_preference == "cpu":
        device = "cpu"
        if verbose:
            logger.info("Whisper device preference forces CPU mode.")
    else:
        cuda_supported = _ctranslate2_cuda_supported(verbose=verbose)
        if device_preference == "cuda" and not cuda_supported and verbose:
            logger.warning("CUDA was requested but is unavailable; falling back to CPU.")
        device = "cuda" if cuda_supported and device_preference != "cpu" else "cpu"

    if compute_preference == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    elif device == "cpu" and compute_preference == "float16":
        compute_type = "int8"
        if verbose:
            logger.warning("float16 compute is not supported on the CPU runtime; using int8 instead.")
    else:
        compute_type = compute_preference

    # Configure GPU memory allocation for torch-based components when available.
    # This is best-effort only: Whisper uses CTranslate2 and can still run on CUDA
    # even when torch is CPU-only.
    if device == "cuda":
        torch_module = get_torch(context="youtube_transcriber:_setup_device_and_compute_type")
        if torch_module is not None:
            try:
                if torch_module.cuda.is_available():
                    gpu_memory_fraction = _gpu_memory_fraction()
                    torch_module.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)
                    if verbose:
                        gpu_name = torch_module.cuda.get_device_name(0)
                        gpu_memory = torch_module.cuda.get_device_properties(0).total_memory / 1024**3
                        logger.info(f"GPU Accelerated: {gpu_name} ({gpu_memory:.1f} GB)")
                        logger.info(
                            f"GPU Memory Allocated: {int(gpu_memory_fraction*100)}% "
                            f"({gpu_memory * gpu_memory_fraction:.1f} GB for maximum performance)"
                        )
                elif verbose:
                    logger.info("PyTorch is CPU-only; using CTranslate2 CUDA backend for Whisper.")
            except Exception as e:
                logger.warning(f"PyTorch CUDA memory tuning skipped: {e}")
        elif verbose:
            logger.info("PyTorch unavailable; using CTranslate2 CUDA backend for Whisper.")
    elif verbose:
        logger.info("No GPU detected. Using CPU.")

    if verbose:
        logger.info("Whisper runtime selected: device=%s | compute_type=%s", device, compute_type)

    return device, compute_type


def _ensure_ffmpeg_on_path(ffmpeg_location: Optional[str], *, context: str) -> None:
    """Prepend an FFmpeg directory to PATH once for subprocess-based backends."""
    if not ffmpeg_location:
        return

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    normalized_entries = {os.path.normcase(entry) for entry in path_entries}
    if os.path.normcase(ffmpeg_location) in normalized_entries:
        return

    os.environ["PATH"] = ffmpeg_location + os.pathsep + current_path if current_path else ffmpeg_location
    logger.info("Added FFmpeg to PATH for %s: %s", context, ffmpeg_location)


def _build_whisper_pipeline(
    model_name: str,
    *,
    device: str,
    compute_type: str,
) -> tuple[Any, Any]:
    """Build a faster-whisper model plus batched inference pipeline."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    base_model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=None,
        num_workers=4 if device == "cpu" else 1,
    )
    return base_model, BatchedInferencePipeline(model=base_model)


def _load_whisper_pipeline_with_fallback(
    model_name: str,
    *,
    device: str,
    compute_type: str,
) -> tuple[Any, Any, str, str]:
    """Build the preferred pipeline and retry on CPU when CUDA libraries are unavailable."""
    try:
        base_model, pipeline = _build_whisper_pipeline(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        if device == "cuda" and _needs_cuda_cpu_fallback(exc):
            logger.warning("CUDA libraries not available (e.g., cublas). Falling back to CPU.")
            device = "cpu"
            compute_type = "int8"
            base_model, pipeline = _build_whisper_pipeline(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        else:
            raise

    return base_model, pipeline, device, compute_type


@dataclass
class _WhisperExecutionState:
    """Mutable faster-whisper execution state shared across retry attempts."""

    model_name: str
    base_model: Any
    pipeline: Any
    device: str
    compute_type: str


def _initialize_whisper_execution(
    model_name: str,
    *,
    config: TranscriptionConfig,
    ffmpeg_location: Optional[str],
    ffmpeg_context: str,
    verbose: bool,
    load_message: str,
    optimized_message: str | None = None,
) -> _WhisperExecutionState:
    """Create shared faster-whisper execution state for a transcription flow."""
    _ensure_ffmpeg_on_path(ffmpeg_location, context=ffmpeg_context)
    device, compute_type = _setup_device_and_compute_type(config=config, verbose=verbose)

    logger.info(load_message)
    if optimized_message:
        logger.info(optimized_message)

    base_model, pipeline, device, compute_type = _load_whisper_pipeline_with_fallback(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    logger.info(f"Transcribing on {device.upper()} with {compute_type}...")
    logger.info("-" * 50)
    return _WhisperExecutionState(
        model_name=model_name,
        base_model=base_model,
        pipeline=pipeline,
        device=device,
        compute_type=compute_type,
    )


def _log_whisper_info(info: Any) -> None:
    """Log the shared metadata emitted by faster-whisper after a run."""
    logger.info(f"Audio duration: {info.duration/60:.1f} minutes")
    logger.info(
        f"Language detected: {info.language} (probability: {info.language_probability:.2f})"
    )
    logger.info("-" * 50)


def _split_segment_text_for_log(text: str, *, max_chars: int = 160) -> list[str]:
    """Split long segment text into sentence-sized log lines for the process pane."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentence_parts:
        sentence_parts = [normalized]

    log_lines: list[str] = []
    for sentence in sentence_parts:
        clause_parts = [sentence]
        if len(sentence) > max_chars:
            split_clauses = [part.strip() for part in re.split(r"(?<=[,;:])\s+", sentence) if part.strip()]
            if split_clauses:
                clause_parts = split_clauses

        for clause in clause_parts:
            if len(clause) <= max_chars:
                log_lines.append(clause)
                continue

            words = clause.split()
            current_words: list[str] = []
            current_length = 0
            for word in words:
                projected_length = current_length + len(word) + (1 if current_words else 0)
                if current_words and projected_length > max_chars:
                    log_lines.append(" ".join(current_words))
                    current_words = [word]
                    current_length = len(word)
                else:
                    current_words.append(word)
                    current_length = projected_length
            if current_words:
                log_lines.append(" ".join(current_words))

    return log_lines or [normalized]


def _iter_segment_log_lines(start_seconds: float, end_seconds: float, text: str) -> list[str]:
    """Format a transcript segment as one or more readable process-log lines."""
    timestamp = f"[{start_seconds/60:05.1f}m -> {end_seconds/60:05.1f}m]"
    text_lines = _split_segment_text_for_log(text)
    if not text_lines:
        return [timestamp]

    lines = [f"{timestamp} {text_lines[0]}"]
    continuation_prefix = " " * len(timestamp) + "   "
    for line in text_lines[1:]:
        lines.append(f"{continuation_prefix}{line}")
    return lines


def _collect_logged_segments(
    segments: Iterable[Any],
    *,
    segment_observer: Callable[[Any, Any], None] | None = None,
    info: Any | None = None,
) -> TranscriptSegments:
    """Convert faster-whisper segments into transcript segments with consistent logging."""
    collected: TranscriptSegments = []
    for segment in segments:
        if segment_observer is not None and info is not None:
            segment_observer(segment, info)

        for log_line in _iter_segment_log_lines(segment.start, segment.end, segment.text):
            logger.info(log_line)
        collected.append(
            make_transcript_segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
        )
    return collected


def _run_whisper_transcription(
    state: _WhisperExecutionState,
    input_source: Any,
    *,
    kwargs: dict[str, Any],
    cpu_recovery_overrides: dict[str, Any] | None = None,
    before_retry: Callable[[], None] | None = None,
    pass_name: str | None = None,
    segment_observer: Callable[[Any, Any], None] | None = None,
) -> tuple[TranscriptSegments, Any, _WhisperExecutionState]:
    """Run faster-whisper using shared execution state with CPU fallback when needed."""
    if pass_name:
        clip_val = kwargs.get("clip_timestamps")
        clip_count = len(clip_val) if isinstance(clip_val, list) else None
        logger.info(
            "Transcription attempt: %s (vad_filter=%s, no_speech_threshold=%s, clip_timestamps=%s)",
            pass_name,
            kwargs.get("vad_filter"),
            kwargs.get("no_speech_threshold"),
            clip_count,
        )

    active_kwargs = dict(kwargs)

    def _execute_current_pipeline() -> tuple[TranscriptSegments, Any]:
        segments, info = state.pipeline.transcribe(input_source, **active_kwargs)
        _log_whisper_info(info)
        segments_data = _collect_logged_segments(
            segments,
            segment_observer=segment_observer,
            info=info,
        )
        return segments_data, info

    while True:
        try:
            segments_data, info = _execute_current_pipeline()
            break
        except Exception as exc:
            if state.device == "cuda" and _is_cuda_out_of_memory_error(exc):
                current_batch_size = max(1, int(active_kwargs.get("batch_size", 1) or 1))
                if current_batch_size > 1:
                    next_batch_size = max(1, current_batch_size // 2)
                    if before_retry is not None:
                        before_retry()
                    logger.warning(
                        "CUDA OOM detected at batch %s. Retrying on CUDA with smaller batch %s.",
                        current_batch_size,
                        next_batch_size,
                    )
                    active_kwargs["batch_size"] = next_batch_size
                    continue

            if state.device == "cuda" and _needs_cuda_cpu_fallback(exc):
                if before_retry is not None:
                    before_retry()

                if cpu_recovery_overrides:
                    logger.warning("GPU issue detected (OOM or missing CUDA libs). Falling back to CPU with INT8 quantization.")
                else:
                    logger.warning("CUDA runtime error detected; retrying transcription on CPU.")

                state.device = "cpu"
                state.compute_type = "int8"
                state.base_model, state.pipeline = _build_whisper_pipeline(
                    state.model_name,
                    device=state.device,
                    compute_type=state.compute_type,
                )
                logger.info(f"Transcribing on {state.device.upper()} with {state.compute_type}...")

                if cpu_recovery_overrides:
                    active_kwargs.update(cpu_recovery_overrides)
                segments_data, info = _execute_current_pipeline()
                if cpu_recovery_overrides:
                    logger.info("Recovery successful with CPU INT8 fallback")
                break

            raise

    # Persist the effective retry settings so later passes inherit the working runtime budget.
    kwargs.clear()
    kwargs.update(active_kwargs)
    return segments_data, info, state


def transcribe_audio(
    audio_file: str,
    ffmpeg_location: Optional[str] = None,
    config: Optional[TranscriptionConfig] = None,
    cleanup_audio_file: bool = True,
) -> tuple[str | None, TranscriptSegments | None]:
    """Transcribe audio file using faster-whisper with GPU acceleration and error recovery.

    Args:
        audio_file: Path to audio file
        ffmpeg_location: Path to FFmpeg directory
        config: TranscriptionConfig with model parameters (uses defaults if None)
        cleanup_audio_file: Remove the audio file after transcription (set False when caller owns it)

    Returns:
        Tuple of (transcript_text, segments_data). Returns (None, None) on error.
        This best-effort contract is intentionally preserved for existing
        YouTube fallback callers; local-file transcription uses typed exceptions.
    """
    if config is None:
        config = get_config().transcription

    torch_module = get_torch(context="youtube_transcriber:transcribe_audio")
    device = "cpu"
    audio_duration = 0.0
    segment_count = 0
    execution_state: _WhisperExecutionState | None = None
    _preprocess_temp: str | None = None

    try:
        import time

        _log_transcription_runtime_config(config, context="YouTube/audio transcription")

        execution_state = _initialize_whisper_execution(
            config.whisper_model,
            config=config,
            ffmpeg_location=ffmpeg_location,
            ffmpeg_context="Whisper",
            verbose=True,
            load_message=(
                f"Loading faster-whisper '{config.whisper_model}' model "
                "(best accuracy, ~3GB download on first run)..."
            ),
            optimized_message="Using optimized CTranslate2 backend with batch processing for 2-4x speedup...",
        )
        device = execution_state.device

        preprocessed_audio_file = _maybe_preprocess_audio_path(
            audio_file,
            ffmpeg_location=ffmpeg_location,
            config=config,
        )
        if preprocessed_audio_file != audio_file:
            _preprocess_temp = preprocessed_audio_file

        start_time = time.time()
        transcribe_kwargs = _build_runtime_transcribe_kwargs(
            config,
            model_name=execution_state.model_name,
            device=execution_state.device,
            compute_type=execution_state.compute_type,
            context="YouTube/audio transcription",
        )

        last_print_time = start_time

        def _observe_segment(segment: Any, info: Any) -> None:
            nonlocal last_print_time
            current_time = time.time()
            audio_duration_local = info.duration
            if current_time - last_print_time >= PROGRESS_DISPLAY_INTERVAL_S:
                elapsed = current_time - start_time
                progress = ((segment.end / audio_duration_local) * 100) if audio_duration_local > 0 else 0
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"[{segment.start/60:.1f}m - {segment.end/60:.1f}m] | "
                    f"Speed: {segment.end/elapsed:.1f}x real-time"
                )
                last_print_time = current_time

            if config.word_timestamps and hasattr(segment, "words") and segment.words:
                for word in segment.words:
                    if hasattr(word, "probability") and word.probability < 0.5:
                        logger.warning(
                            "Low confidence word: '%s' (%.2f) at %.1fs",
                            word.word,
                            word.probability,
                            word.start,
                        )

        segments_data, info, execution_state = _run_whisper_transcription(
            execution_state,
            preprocessed_audio_file,
            kwargs=transcribe_kwargs,
            cpu_recovery_overrides={
                "beam_size": 1,
                "best_of": 1,
                "batch_size": _cpu_fallback_batch_size(config),
            },
            before_retry=(
                lambda: torch_module.cuda.empty_cache()
                if torch_module is not None and torch_module.cuda.is_available()
                else None
            ),
            segment_observer=_observe_segment,
        )
        device = execution_state.device
        audio_duration = info.duration

        if transcription_result_looks_suspicious(segments_data, input_duration=audio_duration):
            logger.warning(
                "Primary transcription output looks suspicious; retrying with stricter anti-hallucination settings."
            )
            retry_kwargs = build_suspicion_retry_kwargs(transcribe_kwargs)
            segments_data, info, execution_state = _run_whisper_transcription(
                execution_state,
                preprocessed_audio_file,
                kwargs=retry_kwargs,
                cpu_recovery_overrides={
                    "beam_size": 1,
                    "best_of": 1,
                    "batch_size": _cpu_fallback_batch_size(config),
                },
                before_retry=(
                    lambda: torch_module.cuda.empty_cache()
                    if torch_module is not None and torch_module.cuda.is_available()
                    else None
                ),
                pass_name="anti-hallucination-retry",
                segment_observer=_observe_segment,
            )
            device = execution_state.device
            audio_duration = info.duration
            if transcription_result_looks_suspicious(segments_data, input_duration=audio_duration):
                logger.warning("Retry output still looks suspicious; rejecting transcript.")
                segments_data = []

        full_transcript, segments_data, removed_count, hallucination_count = _normalize_transcript_segments(
            segments_data,
            clean_fillers=config.clean_filler_words,
            filter_hallucinated=config.filter_hallucinations,
            deduplicate_repetitions=config.deduplicate_repeated_segments,
        )
        if removed_count > 0:
            logger.info("Removed %d repetitive segments from transcript", removed_count)
        if hallucination_count > 0 and not segments_data:
            logger.warning("Only hallucination segments remained after filtering.")
        elif hallucination_count > 0:
            logger.info("Filtered %d hallucination segment(s) from transcript", hallucination_count)
        segment_count = len(segments_data)

        end_time = time.time()
        elapsed_time = end_time - start_time
        speed_factor = audio_duration / elapsed_time if elapsed_time > 0 else 0

        logger.info("-" * 50)
        logger.info("Transcription completed!")
        logger.info(f"Segments processed: {segment_count}")
        logger.info(f"Processing time: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f}s)")
        logger.info(
            f"Speed: {speed_factor:.1f}x real-time "
            f"(processed {audio_duration/60:.1f} min of audio in {elapsed_time/60:.1f} min)"
        )

        # Show GPU memory usage
        if device == "cuda" and torch_module is not None and torch_module.cuda.is_available():
            max_memory = torch_module.cuda.max_memory_allocated(0) / 1024**3
            logger.info(f"Peak GPU memory used: {max_memory:.2f} GB")

        return full_transcript, segments_data

    except Exception as e:
        logger.exception("Error transcribing audio: %s", e)
        return None, None
    finally:
        try:
            if execution_state is not None:
                execution_state.pipeline = None
                execution_state.base_model = None
            if device == "cuda" and torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
                allocated_mb = torch_module.cuda.memory_allocated() / (1024 * 1024)
                if allocated_mb > GPU_MEMORY_WARNING_MB:
                    logger.warning(f"GPU memory not fully released: {allocated_mb:.1f}MB still allocated")
                else:
                    logger.debug(f"GPU memory cache cleared ({allocated_mb:.1f}MB remaining)")
        except Exception as cleanup_exc:
            logger.warning(f"GPU cleanup error (non-fatal): {cleanup_exc}")

        if _preprocess_temp and os.path.exists(_preprocess_temp):
            try:
                os.unlink(_preprocess_temp)
            except OSError:
                pass

        if cleanup_audio_file and audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                logger.info(f"Cleaned up temporary file: {audio_file}")
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temporary file {audio_file}: {cleanup_error}")


def transcribe_local_file(
    file_path: str,
    ffmpeg_location: Optional[str] = None,
    config: Optional[TranscriptionConfig] = None,
    execution_state: _WhisperExecutionState | None = None,
    execution_state_observer: Callable[[_WhisperExecutionState], None] | None = None,
) -> tuple[str, TranscriptSegments]:
    """Transcribe a local audio/video file.

    Args:
        file_path: Path to local audio or video file
        ffmpeg_location: Path to FFmpeg directory
        config: TranscriptionConfig with model parameters (uses defaults if None)
        execution_state: Optional preloaded faster-whisper runtime to reuse
        execution_state_observer: Optional callback receiving the effective runtime state

    Returns:
        Tuple of (transcript_text, segments_data) where:
        - transcript_text: Full transcript as string
        - segments_data: Normalized transcript segments

    Raises:
        FileValidationError: File missing/invalid or too large (audio inputs only)
        TranscriberError: Any other transcription failure
    """
    if config is None:
        config = get_config().transcription
    file_ext = os.path.splitext(file_path)[1].lower()
    torch_module = get_torch(context="youtube_transcriber:transcribe_local_file")

    if torch_module is None:
        logger.warning("torch is unavailable; continuing with non-torch transcription path")

    # Validate file exists
    if not os.path.exists(file_path):
        raise FileValidationError(f"File not found: {file_path}")

    # Validate file size (audio inputs only). Video containers can be very large even when the audio track is small.
    file_size_mb = os.path.getsize(file_path) / (1024**2)
    max_audio_size_mb = _max_audio_size_mb()
    if file_ext not in VIDEO_EXTENSIONS and file_size_mb > max_audio_size_mb:
        raise FileValidationError(
            f"Audio file too large: {file_size_mb:.1f}MB (max: {max_audio_size_mb}MB)"
        )
    if file_ext in VIDEO_EXTENSIONS and file_size_mb > max_audio_size_mb:
        logger.info(
            f"Video file is large ({file_size_mb:.1f}MB) but size limit applies to audio; continuing."
        )

    logger.info(f"Processing local file: {file_path} ({file_size_mb:.1f}MB)")
    logger.info("-" * 50)
    _log_transcription_runtime_config(config, context="Local-file transcription")

    import tempfile
    import time

    temp_paths: list[str] = []
    transcription_input_path = file_path
    ranked_audio_streams: list[AudioStreamCandidate] = []
    selected_audio_index: int | None = None
    best_stream_low_energy = False
    active_execution_state = execution_state

    def _create_temp_wav(
        *,
        audio_index: int | None,
        duration_seconds: int | None = None,
        gain_db: float | None = None,
    ) -> str | None:
        temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_audio_fd)
        try:
            _extract_audio_to_wav(
                file_path,
                temp_audio_path,
                ffmpeg_location,
                audio_index=audio_index,
                duration_seconds=duration_seconds,
                gain_db=gain_db,
            )
        except Exception as exc:  # noqa: BLE001 - surface error in logs, keep going
            logger.debug(f"Audio extraction failed: {exc}")
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass
            return None
        temp_paths.append(temp_audio_path)
        return temp_audio_path

    def _prepare_transcription_input(input_path: str) -> str:
        """Apply optional preprocessing to the active transcription source."""
        prepared_path = _maybe_preprocess_audio_path(
            input_path,
            ffmpeg_location=ffmpeg_location,
            config=config,
            temp_paths=temp_paths,
        )
        if prepared_path != input_path:
            logger.info(f"Using preprocessed audio input: {prepared_path}")
        return prepared_path

    try:
        if file_ext not in VIDEO_EXTENSIONS:
            transcription_input_path = _prepare_transcription_input(file_path)

        # For video containers, proactively rank audio tracks and extract the best candidate.
        if file_ext in VIDEO_EXTENSIONS:
            ranked_audio_streams = _rank_audio_streams_for_transcription(file_path, ffmpeg_location)
            if ranked_audio_streams:
                logger.info("Detected audio streams (ranked by energy):")
                for candidate in ranked_audio_streams:
                    logger.info(f"  - {candidate.describe()}")
                best_stream_low_energy = ranked_audio_streams[0].rms < 0.001
                if best_stream_low_energy:
                    logger.warning(
                        "Audio energy looks extremely low on the best stream (rms=%.4f). "
                        "If you expect speech, the recording may be muted or on a different track.",
                        ranked_audio_streams[0].rms,
                    )
                selected_audio_index = ranked_audio_streams[0].audio_index
            else:
                logger.info("Could not enumerate audio streams; using default audio track (a:0).")
                selected_audio_index = 0

            extracted = _create_temp_wav(audio_index=selected_audio_index)
            if extracted is not None:
                transcription_input_path = _prepare_transcription_input(extracted)
                logger.info(f"Using extracted WAV for transcription (a:{selected_audio_index}): {extracted}")
            else:
                logger.warning("Audio extraction failed; transcribing original video file directly.")

        if active_execution_state is None:
            active_execution_state = _initialize_whisper_execution(
                config.whisper_model,
                config=config,
                ffmpeg_location=ffmpeg_location,
                ffmpeg_context="local transcription",
                verbose=True,
                load_message=f"Loading faster-whisper '{config.whisper_model}' model...",
            )
        else:
            _ensure_ffmpeg_on_path(ffmpeg_location, context="local transcription")
            logger.info(
                "Reusing loaded faster-whisper '%s' model on %s (%s).",
                active_execution_state.model_name,
                active_execution_state.device.upper(),
                active_execution_state.compute_type,
            )
            logger.info("-" * 50)

        # Build transcription kwargs from config and clamp them to the active runtime budget.
        transcribe_kwargs = _build_runtime_transcribe_kwargs(
            config,
            model_name=active_execution_state.model_name,
            device=active_execution_state.device,
            compute_type=active_execution_state.compute_type,
            context="Local-file transcription",
        )

        input_duration = _probe_duration_seconds(transcription_input_path, ffmpeg_location)
        if input_duration is not None:
            logger.info(f"Input duration estimate: {input_duration/60:.1f} minutes")

        def _generate_clip_timestamps(duration_seconds: float, *, chunk_seconds: float = 30.0) -> list[dict[str, float]]:
            clips: list[dict[str, float]] = []
            start = 0.0
            while start < duration_seconds:
                end = min(start + chunk_seconds, duration_seconds)
                if end - start <= 0.0:
                    break
                clips.append({"start": start, "end": end})
                start = end
            return clips

        def _ensure_clip_timestamps_for_long_audio(kwargs: dict[str, Any]) -> None:
            if kwargs.get("vad_filter") is not False:
                return
            if kwargs.get("clip_timestamps"):
                return
            if input_duration is None:
                # faster-whisper requires clip_timestamps when vad_filter=False for long audio.
                logger.warning(
                    "vad_filter=False but duration could not be determined; enabling VAD to avoid faster-whisper error."
                )
                kwargs["vad_filter"] = True
                default_vad_params = transcribe_kwargs.get("vad_parameters")
                kwargs["vad_parameters"] = (
                    dict(default_vad_params) if isinstance(default_vad_params, dict) else {}
                )
                return
            kwargs["clip_timestamps"] = _generate_clip_timestamps(input_duration)

        def _transcribe_with_cuda_fallback(
            pass_name: str, *, kwargs: dict[str, Any]
        ) -> tuple[TranscriptSegments, Any]:
            nonlocal active_execution_state
            _ensure_clip_timestamps_for_long_audio(kwargs)
            assert active_execution_state is not None
            start_time = time.time()
            last_progress_report = start_time - PROGRESS_DISPLAY_INTERVAL_S

            def _observe_segment(segment: Any, info: Any) -> None:
                nonlocal last_progress_report
                current_time = time.time()
                if current_time - last_progress_report < PROGRESS_DISPLAY_INTERVAL_S:
                    return
                audio_duration_local = info.duration
                progress = ((segment.end / audio_duration_local) * 100) if audio_duration_local > 0 else 0.0
                elapsed = max(current_time - start_time, 0.001)
                logger.info(
                    "Local-file progress: %.1f%% | [%.1fs - %.1fs] | Speed: %.1fx real-time",
                    progress,
                    segment.start,
                    segment.end,
                    segment.end / elapsed,
                )
                last_progress_report = current_time

            segments_data, info, active_execution_state = _run_whisper_transcription(
                active_execution_state,
                transcription_input_path,
                kwargs=kwargs,
                cpu_recovery_overrides={
                    "beam_size": 1,
                    "best_of": 1,
                    "batch_size": _cpu_fallback_batch_size(config),
                },
                before_retry=(
                    lambda: torch_module.cuda.empty_cache()
                    if torch_module is not None and torch_module.cuda.is_available()
                    else None
                ),
                pass_name=pass_name,
                segment_observer=_observe_segment,
            )
            return segments_data, info

        def _try_transcribe(pass_name: str, kwargs: dict[str, Any]) -> TranscriptSegments:
            """Attempt transcription and return segments (empty if suspicious)."""
            result, _ = _transcribe_with_cuda_fallback(pass_name, kwargs=kwargs)
            if result and transcription_result_looks_suspicious(result, input_duration=input_duration):
                logger.warning("%s output looks suspicious; continuing with retries.", pass_name)
                return []
            return result

        def _build_relaxed_kwargs() -> dict[str, Any]:
            relaxed_kwargs = dict(transcribe_kwargs)
            relaxed_kwargs["vad_filter"] = True
            relaxed_kwargs["vad_parameters"] = {
                "threshold": min(config.vad_threshold, 0.2),
                "min_speech_duration_ms": min(config.min_speech_duration_ms, 50),
                "max_speech_duration_s": MAX_SPEECH_DURATION_S,
                "min_silence_duration_ms": min(config.min_silence_duration_ms, 500),
                "speech_pad_ms": config.speech_pad_ms,
            }
            relaxed_kwargs["no_speech_threshold"] = min(
                float(relaxed_kwargs.get("no_speech_threshold", 0.3)), 0.1
            )
            relaxed_kwargs["log_prob_threshold"] = -2.0
            return relaxed_kwargs

        def _build_no_vad_kwargs() -> dict[str, Any]:
            base = _build_relaxed_kwargs()
            base["vad_filter"] = False
            base["condition_on_previous_text"] = False
            base.pop("vad_parameters", None)
            return base

        # Strategy sequence: default -> relaxed-vad -> no-vad
        segments_data = _try_transcribe("default", transcribe_kwargs)

        if not segments_data:
            logger.warning("No segments returned. Retrying with relaxed parameters.")
            segments_data = _try_transcribe("relaxed-vad", _build_relaxed_kwargs())

        if not segments_data:
            logger.warning("Still no segments. Retrying without VAD using fixed 30s clips.")
            segments_data = _try_transcribe("no-vad", _build_no_vad_kwargs())

        # If still empty and we have multiple audio streams, try other streams (common in meeting recordings).
        if (
            not segments_data
            and file_ext in VIDEO_EXTENSIONS
            and ranked_audio_streams
            and len(ranked_audio_streams) > 1
        ):
            for candidate in ranked_audio_streams[1:5]:
                logger.warning(
                    "No speech detected on selected audio stream; trying alternate stream %s",
                    candidate.describe(),
                )
                extracted_alt = _create_temp_wav(audio_index=candidate.audio_index)
                if extracted_alt is None:
                    continue
                transcription_input_path = _prepare_transcription_input(extracted_alt)
                input_duration = _probe_duration_seconds(transcription_input_path, ffmpeg_location)

                # Try each strategy on this alternate stream
                segments_data = _try_transcribe("default-alt", transcribe_kwargs)
                if not segments_data:
                    segments_data = _try_transcribe("relaxed-alt", _build_relaxed_kwargs())
                if not segments_data:
                    segments_data = _try_transcribe("no-vad-alt", _build_no_vad_kwargs())
                if segments_data:
                    logger.info("Selected alternate stream for transcription: %s", candidate.describe())
                    break

        if (
            not segments_data
            and file_ext in VIDEO_EXTENSIONS
            and best_stream_low_energy
            and selected_audio_index is not None
        ):
            logger.warning(
                "Audio energy is extremely low; retrying after applying +30dB gain during extraction."
            )
            extracted_gain = _create_temp_wav(audio_index=selected_audio_index, gain_db=30.0)
            if extracted_gain is not None:
                transcription_input_path = _prepare_transcription_input(extracted_gain)
                input_duration = _probe_duration_seconds(transcription_input_path, ffmpeg_location)
                segments_data = _try_transcribe("gain-default", transcribe_kwargs)
                if not segments_data:
                    segments_data = _try_transcribe("gain-relaxed", _build_relaxed_kwargs())

        if not segments_data:
            logger.warning(
                "No speech detected after retries. If this file has speech, try disabling VAD "
                "(set transcription.vad_filter=false in config.json) or inspect audio tracks with ffmpeg/ffprobe."
            )

        full_transcript, segments_data, removed_count, hallucination_count = _normalize_transcript_segments(
            segments_data,
            clean_fillers=config.clean_filler_words,
            filter_hallucinated=config.filter_hallucinations,
            deduplicate_repetitions=config.deduplicate_repeated_segments,
        )
        if removed_count > 0:
            logger.info("Removed %d repetitive segments from transcript", removed_count)
        if hallucination_count > 0 and not segments_data:
            logger.warning("Only hallucination segments remained after filtering.")
        logger.info("-" * 50)
        logger.info(f"Transcription complete: {len(segments_data)} segments")

        return full_transcript, segments_data

    except (FileValidationError, TranscriberError):
        raise
    except Exception as exc:
        logger.exception("Error transcribing local file")
        raise TranscriberError(f"Error transcribing local file: {exc}") from exc
    finally:
        if execution_state_observer is not None and active_execution_state is not None:
            execution_state_observer(active_execution_state)
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError:
                pass



def resolve_output_path(candidate: str, *, base_dir: Optional[Path] = None) -> Path:
    """Resolve a user-provided output path within a trusted base directory."""
    base = base_dir or Path.cwd()
    base = base.resolve()

    output_path = Path(candidate).expanduser()
    if not output_path.is_absolute():
        output_path = base / output_path

    resolved = output_path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("Output path must stay within the project directory") from exc

    sanitized_name = sanitize_filename(resolved.name)
    return resolved.with_name(sanitized_name)


def save_transcript(
    transcript: str,
    video_id: str,
    output_file: Optional[str] = None,
    segments_data: TranscriptSegments | None = None,
    output_format: str = "plain"
) -> Optional[str]:
    """Save transcript to a text file with atomic write operation.

    Args:
        transcript: Transcript text to save
        video_id: YouTube video ID (used for default filename)
        output_file: Optional custom output filename
        segments_data: Optional list of segment dicts (start/end/text[/speaker]) for timestamped output
        output_format: Output format - "plain" or "timestamped" (default: "plain")

    Returns:
        Path to saved file or None on error
    """
    if output_file is None:
        output_file = f"{video_id}_transcript.txt"

    try:
        destination = resolve_output_path(output_file)
    except ValueError as exc:
        logger.error(f"Error: {exc}")
        return None

    # Determine what content to write based on format
    if output_format == "timestamped" and segments_data:
        content = format_transcript_with_timestamps(segments_data)
    else:
        content = transcript

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = None
    try:
        # Write to temporary file first (atomic write pattern)
        temp_path = destination.parent / f"{destination.name}.tmp"
        temp_file = str(temp_path)
        with open(temp_file, "w", encoding="utf-8", buffering=8192) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure written to disk

        # Atomic rename
        os.replace(temp_file, destination)
        logger.info(f"Transcript saved to: {destination}")
        if output_format == "timestamped":
            logger.info("Format: Text with timestamps")
        return str(destination)

    except Exception as e:
        logger.error(f"Error saving transcript: {e}")
        # Cleanup temp file if exists
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as cleanup_error:
                logger.warning(f"Could not remove temp file {temp_file}: {cleanup_error}")
        return None


def main() -> None:
    """Main function to orchestrate the transcription process."""
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos with optional timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
  python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ output.txt
  python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --format timestamped
  python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --verbose
        """
    )
    parser.add_argument("youtube_url", help="YouTube video URL")
    parser.add_argument("output_file", nargs="?", default=None, help="Output file path (optional)")
    parser.add_argument(
        "--format",
        choices=["plain", "timestamped"],
        default="plain",
        help="Output format: 'plain' for text only, 'timestamped' for text with timestamps (default: plain)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    # Initialize logging
    setup_logging(verbose=args.verbose)

    # Log warning about missing API if needed
    if not TRANSCRIPT_API_AVAILABLE:
        logger.warning("youtube-transcript-api not installed. Will use audio transcription only.")

    # Validate YouTube URL
    is_valid, error_msg = validate_youtube_url(args.youtube_url)
    if not is_valid:
        logger.error(f"Error: {error_msg}")
        logger.error("Please provide a valid YouTube URL.")
        sys.exit(1)

    # Extract video ID
    video_id = extract_video_id(args.youtube_url)
    if not video_id:
        logger.error("Error: Could not extract video ID from URL")
        sys.exit(1)

    logger.info(f"Processing video ID: {video_id}")
    logger.info("-" * 50)

    # Try to get transcript from YouTube first
    transcript, segments_data = get_youtube_transcript(video_id)

    # If no transcript available, download and transcribe
    if not transcript:
        logger.info("\nNo captions available. Downloading audio and transcribing...")
        logger.info("-" * 50)

        # Find FFmpeg location once for both download and transcription
        ffmpeg_location = find_ffmpeg()

        audio_file = download_audio(
            args.youtube_url, f"temp_audio_{video_id}", ffmpeg_location
        )

        if not audio_file:
            logger.error("Failed to download audio. Exiting.")
            sys.exit(1)

        # transcribe_audio now handles cleanup in its finally block
        transcript, segments_data = transcribe_audio(audio_file, ffmpeg_location)

    # Save transcript
    if transcript:
        logger.info("-" * 50)
        save_transcript(
            transcript,
            video_id,
            args.output_file,
            segments_data=segments_data,
            output_format=args.format
        )
        logger.info("\nTranscription complete!")
    else:
        logger.error("Failed to obtain transcript. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()

