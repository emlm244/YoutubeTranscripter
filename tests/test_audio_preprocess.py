"""Tests for audio preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import wave

import numpy as np

from audio_preprocessor import (
    _noisereduce_available,
    normalize_loudness_array,
    normalize_loudness_file,
    preprocess_file,
    preprocess_array,
    reduce_noise_array,
)


class TestReduceNoiseArray:
    """Test reduce_noise_array."""

    def test_returns_same_shape(self):
        audio = np.random.randn(16000).astype(np.float32)
        fake_module = SimpleNamespace(reduce_noise=lambda **kwargs: kwargs["y"] * 0.5)

        with patch.dict("sys.modules", {"noisereduce": fake_module}):
            result = reduce_noise_array(audio, sample_rate=16000)

        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_fallback_when_noisereduce_missing(self):
        """When noisereduce is not installed, return original audio."""
        audio = np.random.randn(16000).astype(np.float32)
        # Force ImportError by making noisereduce import fail
        with patch.dict("sys.modules", {"noisereduce": None}):
            result = reduce_noise_array(audio)
            # Should return original when import fails
            assert result.shape == audio.shape

    def test_returns_original_on_reducer_failure(self):
        audio = np.random.randn(16000).astype(np.float32)
        fake_module = SimpleNamespace(reduce_noise=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

        with patch.dict("sys.modules", {"noisereduce": fake_module}):
            result = reduce_noise_array(audio, sample_rate=22050)

        assert np.array_equal(result, audio)


class TestPreprocessArray:
    """Test preprocess_array composite function."""

    def test_noop_when_disabled(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = preprocess_array(
            audio,
            noise_reduction=False,
            normalize=False,
        )
        assert np.array_equal(result, audio)

    def test_noise_reduction_only(self):
        audio = np.random.randn(16000).astype(np.float32)
        fake_module = SimpleNamespace(reduce_noise=lambda **kwargs: kwargs["y"] * 0.5)

        with patch.dict("sys.modules", {"noisereduce": fake_module}):
            result = preprocess_array(
                audio,
                noise_reduction=True,
                normalize=False,
            )

        assert result.shape == audio.shape

    def test_normalize_only_uses_round_trip_helper(self, monkeypatch):
        audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        calls = []

        monkeypatch.setattr(
            "audio_preprocessor.normalize_loudness_array",
            lambda arr, **kwargs: calls.append((arr.copy(), kwargs)) or (arr * 0.5),
        )

        result = preprocess_array(audio, noise_reduction=False, normalize=True, target_lufs=-18.0, ffmpeg_cmd="ffmpeg-test")

        assert np.allclose(result, audio * 0.5)
        assert calls[0][1]["target_lufs"] == -18.0
        assert calls[0][1]["ffmpeg_cmd"] == "ffmpeg-test"


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def test_noisereduce_available_handles_import_state():
    with patch.dict("sys.modules", {"noisereduce": SimpleNamespace()}, clear=False):
        assert _noisereduce_available() is True

    with patch.dict("sys.modules", {"noisereduce": None}, clear=False):
        assert _noisereduce_available() is False


def test_normalize_loudness_file_success(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"in")
    calls = []

    monkeypatch.setattr(
        "audio_preprocessor.subprocess.run",
        lambda cmd, check, capture_output: calls.append((cmd, check, capture_output)),
    )

    assert normalize_loudness_file(str(input_path), str(output_path), target_lufs=-18.0, ffmpeg_cmd="ffmpeg-test") is True
    assert calls[0][0][0] == "ffmpeg-test"
    assert "loudnorm=I=-18.0:TP=-1.5:LRA=11" in calls[0][0]


def test_normalize_loudness_file_handles_failures(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"in")

    monkeypatch.setattr(
        "audio_preprocessor.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "ffmpeg")),
    )
    assert normalize_loudness_file(str(input_path), str(output_path)) is False

    monkeypatch.setattr(
        "audio_preprocessor.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ffmpeg")),
    )
    assert normalize_loudness_file(str(input_path), str(output_path)) is False


def test_normalize_loudness_file_uses_temp_output_when_paths_match(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "audio.wav"
    input_path.write_bytes(b"in")
    calls = []

    def _fake_run(cmd, check, capture_output):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"normalized")

    monkeypatch.setattr("audio_preprocessor.subprocess.run", _fake_run)

    assert normalize_loudness_file(str(input_path), str(input_path)) is True
    assert calls
    assert calls[0][calls[0].index("-i") + 1] == str(input_path)
    assert calls[0][-1] != str(input_path)
    assert input_path.read_bytes() == b"normalized"


def test_normalize_loudness_array_returns_original_when_round_trip_fails(monkeypatch):
    audio = np.array([0.1, -0.1], dtype=np.float32)
    monkeypatch.setattr("audio_preprocessor.normalize_loudness_file", lambda *args, **kwargs: False)

    result = normalize_loudness_array(audio)

    assert np.array_equal(result, audio)


def test_normalize_loudness_array_round_trip_success(monkeypatch):
    audio = np.array([0.25, -0.25, 0.5], dtype=np.float32)

    def _fake_normalize(input_path: str, output_path: str, **kwargs):
        _write_wav(Path(output_path), np.array([0.1, -0.1, 0.2], dtype=np.float32))
        return True

    monkeypatch.setattr("audio_preprocessor.normalize_loudness_file", _fake_normalize)

    result = normalize_loudness_array(audio, sample_rate=22050)

    assert result.dtype == np.float32
    assert np.allclose(result[:3], np.array([0.1, -0.1, 0.2], dtype=np.float32), atol=1e-3)


def test_preprocess_file_returns_input_when_all_disabled(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"demo")

    assert preprocess_file(str(input_path), str(tmp_path / "output.wav"), noise_reduction=False, normalize=False) == str(input_path)


def test_preprocess_file_noise_reduction_and_normalize(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_wav(input_path, np.array([0.1, -0.1, 0.2], dtype=np.float32))
    normalize_calls = []

    monkeypatch.setattr("audio_preprocessor._noisereduce_available", lambda: True)
    monkeypatch.setattr("audio_preprocessor.reduce_noise_array", lambda audio, sample_rate: audio * 0.5)
    monkeypatch.setattr(
        "audio_preprocessor.normalize_loudness_file",
        lambda src, dst, **kwargs: normalize_calls.append((src, dst, kwargs)) or True,
    )

    result = preprocess_file(str(input_path), str(output_path), noise_reduction=True, normalize=True, target_lufs=-18.0)

    assert result == str(output_path)
    assert normalize_calls[0][0] == str(output_path)
    assert normalize_calls[0][2]["target_lufs"] == -18.0


def test_preprocess_file_handles_noise_reduction_failure_and_keeps_normalize_source(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_wav(input_path, np.array([0.1, -0.1], dtype=np.float32))
    normalize_calls = []

    monkeypatch.setattr("audio_preprocessor._noisereduce_available", lambda: True)
    monkeypatch.setattr("audio_preprocessor.reduce_noise_array", lambda audio, sample_rate: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        "audio_preprocessor.normalize_loudness_file",
        lambda src, dst, **kwargs: normalize_calls.append((src, dst)) or False,
    )

    result = preprocess_file(str(input_path), str(output_path), noise_reduction=True, normalize=True)

    assert result == str(input_path)
    assert normalize_calls == [(str(input_path), str(output_path))]
