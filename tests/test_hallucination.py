"""Tests for hallucination detection and filtering."""

from __future__ import annotations

from youtube_transcriber import filter_hallucinations, is_hallucination


class TestIsHallucination:
    """Test is_hallucination function."""

    def test_normal_text_not_hallucination(self):
        assert not is_hallucination("The meeting started at 9am.")

    def test_thank_you_for_watching(self):
        assert is_hallucination("Thank you for watching!")

    def test_empty_string(self):
        # Empty string might be hallucination depending on implementation
        result = is_hallucination("")
        assert isinstance(result, bool)

    def test_subscribe_text(self):
        assert is_hallucination("Please subscribe to my channel")

    def test_russian_continuation(self):
        assert is_hallucination("Продолжение следует")

    def test_case_insensitive(self):
        assert is_hallucination("THANK YOU FOR WATCHING")


class TestFilterHallucinations:
    """Test filter_hallucinations function."""

    def test_removes_hallucinations(self):
        segments = ["Hello world.", "Thank you for watching!", "Goodbye."]
        result, count = filter_hallucinations(segments)
        assert count >= 1
        assert "Thank you for watching!" not in result
        assert "Hello world." in result

    def test_no_hallucinations(self):
        segments = ["Hello.", "How are you?"]
        result, count = filter_hallucinations(segments)
        assert count == 0
        assert len(result) == 2

    def test_empty_input(self):
        result, count = filter_hallucinations([])
        assert result == []
        assert count == 0
