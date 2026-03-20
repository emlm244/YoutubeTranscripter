"""Tests for filename sanitization."""

from __future__ import annotations

from youtube_transcriber import sanitize_filename


class TestSanitizeFilename:
    """Test sanitize_filename function."""

    def test_normal_filename(self):
        result = sanitize_filename("my_video")
        assert result == "my_video"

    def test_strips_special_chars(self):
        result = sanitize_filename("my/video\\file:name")
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result

    def test_truncates_long_filename(self):
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 200  # Should be truncated

    def test_empty_string(self):
        result = sanitize_filename("")
        assert isinstance(result, str)

    def test_unicode_characters(self):
        result = sanitize_filename("video_日本語")
        assert isinstance(result, str)
