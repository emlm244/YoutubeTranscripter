"""Audio preprocessing pipeline for improving Whisper transcription accuracy.

Provides optional spectral noise reduction and loudness normalization
applied before audio is sent to Whisper.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


def _noisereduce_available() -> bool:
    """Check if noisereduce library is installed."""
    try:
        import noisereduce  # noqa: F401
        return True
    except ImportError:
        return False


def reduce_noise_array(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    stationary: bool = True,
) -> np.ndarray:
    """Apply spectral noise reduction to a numpy audio array.

    Args:
        audio: 1-D float32 audio array.
        sample_rate: Sample rate in Hz.
        stationary: Use stationary noise reduction (faster, works well for
            constant background noise like fans/hum).

    Returns:
        Noise-reduced audio array (same shape/dtype).
    """
    try:
        import noisereduce as nr
    except ImportError:
        logger.warning("noisereduce not installed; skipping noise reduction")
        return audio

    try:
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=0.75,
        )
        logger.info("Spectral noise reduction applied")
        return reduced.astype(np.float32)
    except Exception as exc:
        logger.warning("Noise reduction failed (%s); using original audio", exc)
        return audio


def normalize_loudness_file(
    input_path: str,
    output_path: str,
    *,
    target_lufs: float = -20.0,
    ffmpeg_cmd: str = "ffmpeg",
) -> bool:
    """Normalize audio loudness using FFmpeg's loudnorm filter (two-pass).

    Args:
        input_path: Path to input WAV file.
        output_path: Path to write normalized WAV file.
        target_lufs: Target integrated loudness in LUFS.
        ffmpeg_cmd: FFmpeg executable name or path.

    Returns:
        True on success, False on failure.
    """
    temp_output_path: str | None = None
    ffmpeg_output_path = output_path
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        fd_out, temp_output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd_out)
        ffmpeg_output_path = temp_output_path

    cmd = [
        ffmpeg_cmd,
        "-y",
        "-v", "error",
        "-nostdin",
        "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        ffmpeg_output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if temp_output_path is not None:
            os.replace(temp_output_path, output_path)
        logger.info("Audio loudness normalized to %.1f LUFS", target_lufs)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("Loudness normalization failed (%s); skipping", exc)
        return False
    finally:
        if temp_output_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(temp_output_path)


def normalize_loudness_array(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    target_lufs: float = -20.0,
    ffmpeg_cmd: str = "ffmpeg",
) -> np.ndarray:
    """Normalize loudness for a numpy audio array via a temp-file round-trip.

    Args:
        audio: 1-D float32 audio array.
        sample_rate: Sample rate in Hz.
        target_lufs: Target integrated loudness.
        ffmpeg_cmd: FFmpeg executable.

    Returns:
        Normalized audio array, or original on failure.
    """
    try:
        import wave
    except ImportError:
        return audio

    fd_in, tmp_in = tempfile.mkstemp(suffix=".wav")
    fd_out, tmp_out = tempfile.mkstemp(suffix=".wav")
    os.close(fd_in)
    os.close(fd_out)

    try:
        # Write numpy array to WAV
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(tmp_in, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

        if not normalize_loudness_file(
            tmp_in, tmp_out, target_lufs=target_lufs, ffmpeg_cmd=ffmpeg_cmd
        ):
            return audio

        # Read normalized WAV back
        with wave.open(tmp_out, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        normalized = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return normalized

    except Exception as exc:
        logger.warning("Loudness normalization round-trip failed (%s)", exc)
        return audio
    finally:
        for p in (tmp_in, tmp_out):
            with contextlib.suppress(OSError):
                os.unlink(p)


def preprocess_file(
    input_path: str,
    output_path: str,
    *,
    noise_reduction: bool = True,
    normalize: bool = True,
    target_lufs: float = -20.0,
    ffmpeg_cmd: str = "ffmpeg",
) -> str:
    """Preprocess an audio file (noise reduction + normalization).

    Reads the file into a numpy array, applies noise reduction, writes back,
    then normalizes loudness. Returns the path to the preprocessed file
    (may be ``output_path`` or ``input_path`` if nothing changed).

    Args:
        input_path: Source audio file (WAV preferred).
        output_path: Destination for preprocessed audio.
        noise_reduction: Apply spectral noise reduction.
        normalize: Apply loudness normalization.
        target_lufs: Target LUFS for normalization.
        ffmpeg_cmd: FFmpeg executable.

    Returns:
        Path to the preprocessed (or original) audio file.
    """
    if not noise_reduction and not normalize:
        return input_path

    changed = False

    if noise_reduction and _noisereduce_available():
        try:
            import wave
            with wave.open(input_path, "rb") as wf:
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            audio = reduce_noise_array(audio, sample_rate=sr)
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
            changed = True
        except Exception as exc:
            logger.warning("File noise reduction failed (%s); skipping", exc)

    if normalize:
        src = output_path if changed else input_path
        if normalize_loudness_file(src, output_path, target_lufs=target_lufs, ffmpeg_cmd=ffmpeg_cmd):
            changed = True

    return output_path if changed else input_path


def preprocess_array(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    noise_reduction: bool = True,
    normalize: bool = True,
    target_lufs: float = -20.0,
    ffmpeg_cmd: str = "ffmpeg",
) -> np.ndarray:
    """Preprocess a numpy audio array (noise reduction + normalization).

    Args:
        audio: 1-D float32 audio array.
        sample_rate: Sample rate in Hz.
        noise_reduction: Apply spectral noise reduction.
        normalize: Apply loudness normalization.
        target_lufs: Target LUFS.
        ffmpeg_cmd: FFmpeg executable.

    Returns:
        Preprocessed audio array.
    """
    if not noise_reduction and not normalize:
        return audio

    if noise_reduction and _noisereduce_available():
        audio = reduce_noise_array(audio, sample_rate=sample_rate)

    if normalize:
        audio = normalize_loudness_array(
            audio, sample_rate=sample_rate,
            target_lufs=target_lufs, ffmpeg_cmd=ffmpeg_cmd,
        )

    return audio
