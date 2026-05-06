"""Tests for URL validation and video ID extraction."""

from __future__ import annotations

from youtube_transcriber import extract_video_id, validate_youtube_url


class TestValidateYoutubeUrl:
    """Test validate_youtube_url function."""

    def test_standard_url(self):
        is_valid, error = validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_valid
        assert error is None

    def test_short_url(self):
        is_valid, error = validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert is_valid
        assert error is None

    def test_rejects_youtube_url_without_video_id(self):
        is_valid, error = validate_youtube_url("https://www.youtube.com/watch?feature=share")
        assert not is_valid
        assert error == "No valid video ID found in URL"

    def test_rejects_bare_video_id_input(self):
        is_valid, error = validate_youtube_url("dQw4w9WgXcQ")
        assert not is_valid
        assert error == "URL must start with http:// or https://"

    def test_empty_string(self):
        is_valid, error = validate_youtube_url("")
        assert not is_valid
        assert error

    def test_non_youtube_url(self):
        is_valid, error = validate_youtube_url("https://example.com/video")
        assert not is_valid
        assert error == "Invalid domain: example.com. Expected a YouTube URL."

    def test_invalid_url(self):
        is_valid, error = validate_youtube_url("not a url")
        assert not is_valid
        assert error == "URL must start with http:// or https://"


class TestExtractVideoId:
    """Test extract_video_id function."""

    def test_standard_url(self):
        vid = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_short_url(self):
        vid = extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_url_with_params(self):
        vid = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120")
        assert vid == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        vid = extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share")
        assert vid == "dQw4w9WgXcQ"

    def test_embed_url(self):
        vid = extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_invalid_url(self):
        vid = extract_video_id("not a url")
        assert vid is None

    def test_invalid_youtube_path(self):
        vid = extract_video_id("https://www.youtube.com/watch?feature=share")
        assert vid is None

    def test_empty_string(self):
        vid = extract_video_id("")
        assert vid is None
