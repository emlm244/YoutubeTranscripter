"""Tests for timestamp and SRT formatting functions."""

from __future__ import annotations

from youtube_transcriber import (
    format_srt_timestamp,
    format_transcript_as_json,
    format_transcript_as_srt,
    format_transcript_with_timestamps,
)


class TestFormatSrtTimestamp:
    """Test format_srt_timestamp."""

    def test_zero(self):
        assert format_srt_timestamp(0.0) == "00:00:00,000"

    def test_seconds_and_millis(self):
        assert format_srt_timestamp(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert format_srt_timestamp(90.0) == "00:01:30,000"

    def test_hours(self):
        assert format_srt_timestamp(3661.123) == "01:01:01,123"

    def test_large_time(self):
        result = format_srt_timestamp(7200.0)
        assert result.startswith("02:00:00")


class TestFormatTranscriptWithTimestamps:
    """Test format_transcript_with_timestamps."""

    def test_basic_segments(self, sample_segments):
        result = format_transcript_with_timestamps(sample_segments)
        assert "Hello world" in result
        assert "[" in result  # Should contain timestamps

    def test_empty_segments(self):
        result = format_transcript_with_timestamps([])
        assert result == ""


class TestFormatTranscriptAsSrt:
    """Test SRT format output."""

    def test_produces_srt(self, sample_segments):
        result = format_transcript_as_srt(sample_segments)
        assert "1\n" in result
        assert "-->" in result
        assert "Hello world" in result

    def test_empty_segments(self):
        result = format_transcript_as_srt([])
        assert result.strip() == ""


class TestFormatTranscriptAsJson:
    """Test JSON format output."""

    def test_produces_valid_json(self, sample_segments):
        import json
        result = format_transcript_as_json(sample_segments)
        data = json.loads(result)
        assert "segments" in data or isinstance(data, list)

    def test_empty_segments(self):
        import json
        result = format_transcript_as_json([])
        data = json.loads(result)
        assert isinstance(data, (list, dict))
