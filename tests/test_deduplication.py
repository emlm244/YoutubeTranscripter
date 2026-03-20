"""Tests for segment deduplication."""

from __future__ import annotations

from youtube_transcriber import deduplicate_segments


class TestDeduplicateSegments:
    """Test deduplicate_segments function."""

    def test_no_duplicates(self):
        segments = ["Hello.", "World.", "Goodbye."]
        result, count = deduplicate_segments(segments)
        assert count == 0
        assert len(result) == 3

    def test_removes_repetitive_segments(self):
        segments = [
            "Hello.",
            "Hello.",
            "Hello.",
            "Hello.",
            "World.",
        ]
        result, count = deduplicate_segments(segments)
        assert count > 0
        assert "World." in result

    def test_empty_input(self):
        result, count = deduplicate_segments([])
        assert result == []
        assert count == 0

    def test_single_segment(self):
        result, count = deduplicate_segments(["Hello."])
        assert result == ["Hello."]
        assert count == 0

    def test_with_return_indices(self):
        segments = ["A.", "A.", "A.", "A.", "B."]
        result, count, indices = deduplicate_segments(
            segments, return_indices=True
        )
        assert isinstance(indices, list)
        assert "B." in result
