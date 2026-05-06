"""Tests for filename sanitization."""

from __future__ import annotations

import youtube_transcriber as yt

from youtube_transcriber import sanitize_filename


class TestSanitizeFilename:
    """Test sanitize_filename function."""

    def test_normal_filename(self):
        result = sanitize_filename("my_video")
        assert result == "my_video"

    def test_strips_special_chars(self, monkeypatch):
        monkeypatch.setattr(yt.sys, "platform", "win32")
        result = sanitize_filename("file:nametest")
        assert result == "filenametest"

    def test_truncates_long_filename(self):
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 200  # Should be truncated

    def test_empty_string(self):
        result = sanitize_filename("")
        assert result == "untitled_video"

    def test_unicode_characters(self):
        expected = "video_\u65e5\u672c\u8a9e"
        result = sanitize_filename(expected)
        assert result == expected

    def test_windows_reserved_basename_with_extension(self, monkeypatch):
        monkeypatch.setattr(yt.sys, "platform", "win32")
        result = sanitize_filename("CON.txt")
        assert result == "_CON.txt"

    def test_windows_trailing_dot_and_space_are_removed(self, monkeypatch):
        monkeypatch.setattr(yt.sys, "platform", "win32")
        result = sanitize_filename("meeting notes .txt. ")
        assert result == "meeting notes.txt"
