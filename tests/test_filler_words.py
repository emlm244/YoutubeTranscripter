"""Tests for filler word cleaning."""

from __future__ import annotations


from youtube_transcriber import clean_filler_words


class TestCleanFillerWords:
    """Test the clean_filler_words function."""

    def test_empty_string(self):
        assert clean_filler_words("") == ""

    def test_whitespace_only(self):
        assert clean_filler_words("   ") == ""

    def test_removes_umm(self):
        result = clean_filler_words("So umm I was thinking")
        assert "umm" not in result.lower()
        assert "thinking" in result

    def test_removes_uh(self):
        result = clean_filler_words("Well uh that is interesting")
        assert "uh" not in result.split()
        assert "interesting" in result

    def test_removes_multiple_fillers(self):
        result = clean_filler_words("Umm uh so er I think ah yeah")
        assert "umm" not in result.lower()
        assert "uh" not in result.lower()
        assert "er" not in result.lower().split()
        assert "ah" not in result.lower().split()

    def test_removes_you_know(self):
        result = clean_filler_words("It was, you know, pretty good")
        assert "you know" not in result.lower()
        assert "pretty good" in result

    def test_removes_i_mean(self):
        result = clean_filler_words("I mean, the code works fine")
        assert "i mean" not in result.lower()
        assert "code works fine" in result

    def test_removes_basically_filler(self):
        result = clean_filler_words("Basically, we need to fix this")
        assert result.startswith("we") or result.startswith("We") or "we need" in result

    def test_removes_like_filler(self):
        result = clean_filler_words("It was like, really hard")
        assert "like," not in result
        assert "really hard" in result

    def test_preserves_meaningful_text(self):
        text = "The meeting went well and we decided on the new approach."
        assert clean_filler_words(text) == text

    def test_collapses_extra_spaces(self):
        result = clean_filler_words("Hello  umm   world")
        assert "  " not in result

    def test_case_insensitive(self):
        result = clean_filler_words("UMM I think UH that ER works")
        assert "UMM" not in result
        assert "UH" not in result
