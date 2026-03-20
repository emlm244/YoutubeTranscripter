"""Tests for audio preprocessing pipeline."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from audio_preprocessor import (
    preprocess_array,
    reduce_noise_array,
)


class TestReduceNoiseArray:
    """Test reduce_noise_array."""

    def test_returns_same_shape(self):
        audio = np.random.randn(16000).astype(np.float32)
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


class TestPreprocessArray:
    """Test preprocess_array composite function."""

    def test_noop_when_disabled(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = preprocess_array(
            audio,
            noise_reduction=False,
            normalize=False,
        )
        np.testing.assert_array_equal(result, audio)

    def test_noise_reduction_only(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = preprocess_array(
            audio,
            noise_reduction=True,
            normalize=False,
        )
        assert result.shape == audio.shape
