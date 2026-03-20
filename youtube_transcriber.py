"""Core transcription functions used by the GUI and CLI.

This module wraps YouTube caption fetching, yt-dlp audio extraction, and local
faster-whisper transcription with optional post-processing.
"""

from __future__ import annotations

import argparse
import array
import glob
import json
import logging
import math
import os
import platform
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
from app_paths import get_ffmpeg_search_roots, get_log_path
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


def _ctranslate2_cuda_supported(*, verbose: bool = False) -> bool:
    """Check whether faster-whisper's CTranslate2 backend can use CUDA."""
    if platform.system() == "Windows" and not ensure_cuda12_runtime_on_windows():
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
        except Exception:
            pass

    # Fall back to CTranslate2-level detection if torch is CPU-only.
    try:
        import ctranslate2

        device_count = ctranslate2.get_cuda_device_count()
        if device_count > 0:
            suffix = "device" if device_count == 1 else "devices"
            return True, f"CTranslate2 CUDA ({device_count} {suffix})"
    except Exception:
        pass

    return True, "CTranslate2 CUDA"


def get_whisper_device_and_compute_type(*, verbose: bool = True) -> tuple[str, str]:
    """Public wrapper used by both CLI and GUI transcription flows."""
    return _setup_device_and_compute_type(verbose=verbose)


def ensure_cuda12_runtime_on_windows() -> bool:
    """Ensure CUDA 12 runtime DLLs are discoverable for CTranslate2 on Windows.

    CTranslate2 GPU wheels currently depend on CUDA 12.x libraries (notably
    ``cublas64_12.dll``). PyTorch can be built against a different CUDA major
    version (e.g., CUDA 13) and does not satisfy this dependency.

    Returns:
        True when ``cublas64_12.dll`` can be loaded, False otherwise.
    """
    if platform.system() != "Windows":
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
        try:
            add_dll_directory(directory_str)
        except OSError:
            continue
        if directory_str not in path_entries:
            os.environ["PATH"] = directory_str + os.pathsep + os.environ.get("PATH", "")

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
PROGRESS_DISPLAY_INTERVAL_S = 5.0
GPU_MEMORY_WARNING_MB = 100
MAX_SPEECH_DURATION_S = 30.0

# Common video container extensions (audio-only size limit should not apply)
VIDEO_EXTENSIONS: set[str] = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}

# FFmpeg path cache (avoids slow recursive directory walk on each call)
_cached_ffmpeg_path: Optional[str] = None
_ffmpeg_cache_checked = False


def _gpu_memory_fraction() -> float:
    return get_config().gpu_memory_fraction


def _max_audio_size_mb() -> int:
    return get_config().max_audio_size_mb


def _max_filename_length() -> int:
    return get_config().max_filename_length


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

    if platform.system() == "Windows":
        # Windows reserved characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)
        # Windows reserved names (complete list)
        reserved = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        ]
        if filename.upper() in reserved:
            filename = f"_{filename}"
    else:
        # Unix-like systems (remove / and null)
        filename = re.sub(r"[/\x00]", "", filename)

    # Ensure filename is not empty after sanitization
    if not filename or filename.strip() == "":
        filename = "untitled_video"

    # Limit length
    return filename[:max_length].strip()


def _normalise_host(netloc: str) -> str:
    """Extract hostname portion without port."""
    return netloc.split(":")[0].lower()


def validate_youtube_url(url: str) -> tuple[bool, str | None]:
    """Validate that the URL is a legitimate YouTube URL.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parsed = urllib.parse.urlparse(url)
        host = _normalise_host(parsed.netloc)

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

        # Check for video ID
        if host == "youtu.be":
            return True, None
        elif "v=" in parsed.query or "/watch" in parsed.path or "/shorts/" in parsed.path:
            return True, None
        else:
            return False, "No video ID found in URL"
    except Exception as e:
        return False, f"Invalid URL format: {e}"


def check_dependencies() -> tuple[bool, list[str]]:
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
    if platform.system() == "Windows" and ffmpeg_location:
        candidate = Path(ffmpeg_location) / f"{name}.exe"
        if candidate.exists():
            return str(candidate)
    return name


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
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        logger.debug("ffprobe not found; skipping audio stream enumeration")
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
        completed = subprocess.run(cmd, check=True, capture_output=True)
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
    subprocess.run(cmd, check=True, capture_output=True)


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
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    output = (completed.stdout or "").strip()
    try:
        return float(output)
    except ValueError:
        return None


def _probe_duration_seconds(file_path: str, ffmpeg_location: Optional[str]) -> float | None:
    return _wav_duration_seconds(file_path) or _ffprobe_duration_seconds(file_path, ffmpeg_location)


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL.

    Args:
        url: YouTube URL or video ID

    Returns:
        Video ID string or None if not found
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",  # YouTube Shorts format
        r"^([0-9A-Za-z_-]{11})$",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

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
    "goodbye",
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


def is_hallucination(text: str) -> bool:
    """Check if text is a known Whisper hallucination phrase.

    Args:
        text: The transcribed text to check

    Returns:
        True if the text appears to be a hallucination
    """
    if not text:
        return False

    cleaned = text.strip().lower()

    # Empty text after cleaning
    if not cleaned:
        return True

    # Check for known hallucination phrases
    for phrase in HALLUCINATION_PHRASES:
        # Match if the text is mostly just the hallucination phrase
        if phrase in cleaned and len(cleaned) < len(phrase) + 15:
            return True

    return False


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
) -> tuple[str, TranscriptSegments, int, int]:
    """Normalize, filter, and deduplicate transcript segments."""
    normalized_segments = list(segments_data)
    hallucination_count = 0

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
    deduplicated_texts, removed_count, kept_indices = deduplicate_segments(
        segment_texts,
        return_indices=True,
    )
    if removed_count > 0:
        normalized_segments = [normalized_segments[index] for index in kept_indices]

    normalized_segments = [
        replace_segment_text(segment, deduplicated_texts[index])
        for index, segment in enumerate(normalized_segments)
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
        Path to downloaded audio file or None on error

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
                )
            elif "Video unavailable" in error_msg or "Private video" in error_msg:
                raise AudioDownloadError(
                    f"Video is unavailable or private: {error_msg}"
                )
            else:
                raise AudioDownloadError(f"Download failed: {error_msg}")

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
        logger.error(f"Unexpected error downloading audio: {e}")
        import traceback
        traceback.print_exc()
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


def _setup_device_and_compute_type(*, verbose: bool = True) -> tuple[str, str]:
    """Determine device and compute type with GPU fallback.

    Checks CUDA availability, CUDA 12 runtime on Windows, and configures
    GPU memory allocation. Falls back to CPU with INT8 if any step fails.

    Args:
        verbose: If True, log detailed GPU information.

    Returns:
        Tuple of (device, compute_type) where device is "cuda" or "cpu"
        and compute_type is "float16" (GPU) or "int8" (CPU).
    """
    device = "cuda" if _ctranslate2_cuda_supported(verbose=verbose) else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

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

    return device, compute_type


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
    """
    if config is None:
        config = get_config().transcription

    torch_module = get_torch(context="youtube_transcriber:transcribe_audio")
    device = "cpu"
    compute_type = "int8"
    audio_duration = 0.0
    segment_count = 0
    base_model: Any = None
    model: Any = None
    _preprocess_temp: str | None = None

    try:
        import time

        if ffmpeg_location:
            current_path = os.environ.get("PATH", "")
            path_entries = current_path.split(os.pathsep) if current_path else []
            if ffmpeg_location not in path_entries:
                os.environ["PATH"] = ffmpeg_location + os.pathsep + current_path
                logger.info(f"Added FFmpeg to PATH for Whisper: {ffmpeg_location}")

        from faster_whisper import BatchedInferencePipeline, WhisperModel

        def _build_pipeline(current_device: str, current_compute_type: str) -> tuple[Any, Any]:
            base_model = WhisperModel(
                config.whisper_model,
                device=current_device,
                compute_type=current_compute_type,
                download_root=None,
                num_workers=4 if current_device == "cpu" else 1,
            )
            return base_model, BatchedInferencePipeline(model=base_model)

        device, compute_type = _setup_device_and_compute_type(verbose=True)

        logger.info(
            f"Loading faster-whisper '{config.whisper_model}' model (best accuracy, ~3GB download on first run)..."
        )
        logger.info(
            "Using optimized CTranslate2 backend with batch processing for 2-4x speedup..."
        )

        try:
            base_model, model = _build_pipeline(device, compute_type)
        except Exception as exc:
            if device == "cuda" and _needs_cuda_cpu_fallback(exc):
                logger.warning("CUDA libraries not available (e.g., cublas). Falling back to CPU.")
                device = "cpu"
                compute_type = "int8"
                base_model, model = _build_pipeline(device, compute_type)
            else:
                raise

        logger.info(f"Transcribing on {device.upper()} with {compute_type}...")
        logger.info("-" * 50)

        preprocessed_audio_file = audio_file
        if config.noise_reduction_enabled or config.normalize_audio:
            import tempfile as _tmpmod

            _pp_fd, _pp_path = _tmpmod.mkstemp(suffix=".wav")
            os.close(_pp_fd)
            ffmpeg_cmd = _ffmpeg_executable("ffmpeg", ffmpeg_location)
            result_path = preprocess_file(
                audio_file,
                _pp_path,
                noise_reduction=config.noise_reduction_enabled,
                normalize=config.normalize_audio,
                ffmpeg_cmd=ffmpeg_cmd,
            )
            if result_path != audio_file:
                preprocessed_audio_file = result_path
                _preprocess_temp = _pp_path
                logger.info("Audio preprocessing applied")
            else:
                try:
                    os.unlink(_pp_path)
                except OSError:
                    pass

        start_time = time.time()
        transcribe_kwargs = _build_transcribe_kwargs(config)

        try:
            segments, info = model.transcribe(preprocessed_audio_file, **transcribe_kwargs)
        except RuntimeError as e:
            if device == "cuda" and _needs_cuda_cpu_fallback(e):
                logger.warning("GPU issue detected (OOM or missing CUDA libs). Falling back to CPU with INT8 quantization.")
                if torch_module is not None and torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                device = "cpu"
                compute_type = "int8"
                base_model, model = _build_pipeline(device, compute_type)

                recovery_kwargs = _build_transcribe_kwargs(config)
                recovery_kwargs.update(
                    beam_size=1,
                    best_of=1,
                    batch_size=8,
                )
                segments, info = model.transcribe(preprocessed_audio_file, **recovery_kwargs)
                logger.info("Recovery successful with CPU INT8 fallback")
            else:
                raise

        audio_duration = info.duration
        logger.info(f"Audio duration: {audio_duration/60:.1f} minutes")
        logger.info(
            f"Language detected: {info.language} (probability: {info.language_probability:.2f})"
        )
        logger.info("-" * 50)

        segments_data: TranscriptSegments = []
        last_print_time = start_time

        for segment in segments:
            current_time = time.time()
            if current_time - last_print_time >= PROGRESS_DISPLAY_INTERVAL_S:
                elapsed = current_time - start_time
                progress = ((segment.end / audio_duration) * 100) if audio_duration > 0 else 0
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"[{segment.start/60:.1f}m - {segment.end/60:.1f}m] | "
                    f"Speed: {segment.end/elapsed:.1f}x real-time"
                )
                last_print_time = current_time

            timestamp = f"[{segment.start/60:05.1f}m -> {segment.end/60:05.1f}m]"
            logger.info(f"{timestamp} {segment.text.strip()}")

            if config.word_timestamps and hasattr(segment, "words") and segment.words:
                for word in segment.words:
                    if hasattr(word, "probability") and word.probability < 0.5:
                        logger.warning(
                            "Low confidence word: '%s' (%.2f) at %.1fs",
                            word.word,
                            word.probability,
                            word.start,
                        )

            segments_data.append(
                make_transcript_segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
            )

        full_transcript, segments_data, removed_count, _ = _normalize_transcript_segments(
            segments_data,
            clean_fillers=config.clean_filler_words,
            filter_hallucinated=False,
        )
        if removed_count > 0:
            logger.info("Removed %d repetitive segments from transcript", removed_count)
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
            model = None
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
) -> tuple[str, TranscriptSegments]:
    """Transcribe a local audio/video file.

    Args:
        file_path: Path to local audio or video file
        ffmpeg_location: Path to FFmpeg directory
        config: TranscriptionConfig with model parameters (uses defaults if None)

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

    if get_torch(context="youtube_transcriber:transcribe_local_file") is None:
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

    import tempfile

    temp_paths: list[str] = []
    transcription_input_path = file_path
    ranked_audio_streams: list[AudioStreamCandidate] = []
    selected_audio_index: int | None = None
    best_stream_low_energy = False

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

    try:
        # Add FFmpeg to PATH if found
        if ffmpeg_location:
            current_path = os.environ.get("PATH", "")
            path_entries = current_path.split(os.pathsep) if current_path else []
            if ffmpeg_location not in path_entries:
                os.environ["PATH"] = ffmpeg_location + os.pathsep + current_path
                logger.info(f"Added FFmpeg to PATH: {ffmpeg_location}")

        # For video containers, proactively extract the most "active" audio stream to WAV.
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
                transcription_input_path = extracted
                logger.info(f"Using extracted WAV for transcription (a:{selected_audio_index}): {extracted}")
            else:
                logger.warning("Audio extraction failed; transcribing original video file directly.")

        # Setup device and compute type with automatic GPU fallback
        device, compute_type = _setup_device_and_compute_type(verbose=True)

        from faster_whisper import BatchedInferencePipeline, WhisperModel

        # Load Whisper model
        logger.info(f"Loading faster-whisper '{config.whisper_model}' model...")
        try:
            base_model = WhisperModel(
                config.whisper_model,  # Use configured model
                device=device,
                compute_type=compute_type,
                download_root=None,
                num_workers=4 if device == "cpu" else 1,
            )
        except Exception as exc:
            if device == "cuda" and _needs_cuda_cpu_fallback(exc):
                logger.warning("CUDA libraries not available (e.g., cublas). Falling back to CPU.")
                device = "cpu"
                compute_type = "int8"
                base_model = WhisperModel(
                    config.whisper_model,
                    device=device,
                    compute_type=compute_type,
                    download_root=None,
                    num_workers=4,
                )
            else:
                raise
        model = BatchedInferencePipeline(model=base_model)

        # Transcribe the file
        logger.info(f"Transcribing on {device.upper()} with {compute_type}...")
        logger.info("-" * 50)

        # Build transcription kwargs from config (centralized parameters)
        transcribe_kwargs = _build_transcribe_kwargs(config)

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

        def _is_suspicious_transcription_result(segments: TranscriptSegments) -> bool:
            if not segments:
                return True
            texts = [
                seg.get("text", "").strip()
                for seg in segments
                if isinstance(seg.get("text", ""), str)
            ]
            texts = [text for text in texts if text]
            if not texts:
                return True

            normalised = [text.lower() for text in texts]
            unique_texts = set(normalised)

            # Strong signal for hallucination/loop: the exact same short text repeated across many chunks.
            if len(texts) >= 20 and len(unique_texts) <= 1:
                only = next(iter(unique_texts))
                if len(only) <= 60:
                    return True

            non_hallucinated = [text for text in texts if not is_hallucination(text)]
            if not non_hallucinated and len(texts) >= 5:
                return True

            if input_duration is not None and input_duration >= 180:
                word_count = len(" ".join(non_hallucinated or texts).split())
                if word_count < 20:
                    return True

            return False

        def _collect_segments(segments: Any) -> TranscriptSegments:
            collected: TranscriptSegments = []
            for segment in segments:
                timestamp = f"[{segment.start/60:05.1f}m -> {segment.end/60:05.1f}m]"
                logger.info(f"{timestamp} {segment.text.strip()}")
                collected.append(
                    make_transcript_segment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                    )
                )
            return collected

        def _transcribe_with_cuda_fallback(
            pass_name: str, *, kwargs: dict[str, Any]
        ) -> tuple[TranscriptSegments, Any]:
            nonlocal device, compute_type, base_model, model
            _ensure_clip_timestamps_for_long_audio(kwargs)
            clip_val = kwargs.get("clip_timestamps")
            clip_count = len(clip_val) if isinstance(clip_val, list) else None
            logger.info(
                "Transcription attempt: %s (vad_filter=%s, no_speech_threshold=%s, clip_timestamps=%s)",
                pass_name,
                kwargs.get("vad_filter"),
                kwargs.get("no_speech_threshold"),
                clip_count,
            )
            try:
                segments, info = model.transcribe(transcription_input_path, **kwargs)
            except Exception as exc:
                if device == "cuda" and _needs_cuda_cpu_fallback(exc):
                    logger.warning("CUDA runtime error detected; retrying transcription on CPU.")
                    device = "cpu"
                    compute_type = "int8"
                    base_model = WhisperModel(
                        config.whisper_model,
                        device=device,
                        compute_type=compute_type,
                        download_root=None,
                        num_workers=4,
                    )
                    model = BatchedInferencePipeline(model=base_model)
                    logger.info(f"Transcribing on {device.upper()} with {compute_type}...")
                    segments, info = model.transcribe(transcription_input_path, **kwargs)
                else:
                    raise

            logger.info(f"Audio duration: {info.duration/60:.1f} minutes")
            logger.info(
                f"Language detected: {info.language} (probability: {info.language_probability:.2f})"
            )
            logger.info("-" * 50)
            return _collect_segments(segments), info

        def _try_transcribe(pass_name: str, kwargs: dict[str, Any]) -> TranscriptSegments:
            """Attempt transcription and return segments (empty if suspicious)."""
            result, _ = _transcribe_with_cuda_fallback(pass_name, kwargs=kwargs)
            if result and _is_suspicious_transcription_result(result):
                logger.warning("%s output looks suspicious; continuing with retries.", pass_name)
                return []
            return result

        # Build relaxed VAD kwargs once for reuse
        relaxed_kwargs: dict[str, Any] | None = None

        def _get_relaxed_kwargs() -> dict[str, Any]:
            nonlocal relaxed_kwargs
            if relaxed_kwargs is None:
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
            base = dict(_get_relaxed_kwargs())
            base["vad_filter"] = False
            base["condition_on_previous_text"] = False
            base.pop("vad_parameters", None)
            return base

        # Strategy sequence: default -> relaxed-vad -> no-vad
        segments_data = _try_transcribe("default", transcribe_kwargs)

        if not segments_data:
            logger.warning("No segments returned. Retrying with relaxed parameters.")
            segments_data = _try_transcribe("relaxed-vad", _get_relaxed_kwargs())

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
                transcription_input_path = extracted_alt
                input_duration = _probe_duration_seconds(transcription_input_path, ffmpeg_location)

                # Try each strategy on this alternate stream
                segments_data = _try_transcribe("default-alt", transcribe_kwargs)
                if not segments_data:
                    segments_data = _try_transcribe("relaxed-alt", _get_relaxed_kwargs())
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
                transcription_input_path = extracted_gain
                input_duration = _probe_duration_seconds(transcription_input_path, ffmpeg_location)
                segments_data = _try_transcribe("gain-default", transcribe_kwargs)
                if not segments_data:
                    segments_data = _try_transcribe("gain-relaxed", _get_relaxed_kwargs())

        if not segments_data:
            logger.warning(
                "No speech detected after retries. If this file has speech, try disabling VAD "
                "(set transcription.vad_filter=false in config.json) or inspect audio tracks with ffmpeg/ffprobe."
            )

        full_transcript, segments_data, removed_count, hallucination_count = _normalize_transcript_segments(
            segments_data,
            clean_fillers=config.clean_filler_words,
            filter_hallucinated=True,
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

