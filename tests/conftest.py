"""Shared fixtures and mocks for test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_segments() -> list[dict]:
    """Return a typical list of transcript segment dicts."""
    return [
        {"start": 0.0, "end": 2.5, "text": "Hello world."},
        {"start": 2.5, "end": 5.0, "text": "This is a test."},
        {"start": 5.0, "end": 8.0, "text": "Thank you for listening."},
    ]


@pytest.fixture
def mock_whisper_model():
    """Return a mock WhisperModel that yields fake segments."""
    model = MagicMock()

    class FakeSegment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text
            self.words = []

    class FakeInfo:
        duration = 10.0
        language = "en"
        language_probability = 0.99

    segments = [
        FakeSegment(0.0, 2.5, "Hello world."),
        FakeSegment(2.5, 5.0, "This is a test."),
    ]
    model.transcribe.return_value = (iter(segments), FakeInfo())
    return model
