from __future__ import annotations

from transcript_types import (
    TranscriptSegment,
    _coerce_float,
    coerce_transcript_segment,
    coerce_transcript_segments,
    make_transcript_segment,
    replace_segment_text,
)


def test_coerce_float_handles_supported_and_invalid_values():
    assert _coerce_float(3) == 3.0
    assert _coerce_float(2.5) == 2.5
    assert _coerce_float("4.25") == 4.25
    assert _coerce_float("not-a-number") == 0.0
    assert _coerce_float(object()) == 0.0


def test_make_transcript_segment_strips_text_and_preserves_speaker():
    assert make_transcript_segment(start=1, end=2, text="  hello world  ", speaker="Speaker 1") == {
        "start": 1.0,
        "end": 2.0,
        "text": "hello world",
        "speaker": "Speaker 1",
    }


def test_make_transcript_segment_omits_blank_speaker():
    assert make_transcript_segment(start=0, end=1, text=" hi ", speaker="") == {
        "start": 0.0,
        "end": 1.0,
        "text": "hi",
    }


def test_replace_segment_text_returns_copy_and_keeps_speaker():
    original: TranscriptSegment = {"start": 0.0, "end": 1.5, "text": "before", "speaker": "Alex"}

    updated = replace_segment_text(original, "after")

    assert updated == {"start": 0.0, "end": 1.5, "text": "after", "speaker": "Alex"}
    assert original == {"start": 0.0, "end": 1.5, "text": "before", "speaker": "Alex"}


def test_replace_segment_text_handles_missing_speaker():
    original: TranscriptSegment = {"start": 2.0, "end": 3.0, "text": "before"}

    assert replace_segment_text(original, "after") == {
        "start": 2.0,
        "end": 3.0,
        "text": "after",
    }


def test_coerce_transcript_segment_normalizes_values():
    segment = {
        "start": "1.25",
        "end": 2,
        "text": 123,
        "speaker": "  ",
    }

    assert coerce_transcript_segment(segment) == {
        "start": 1.25,
        "end": 2.0,
        "text": "123",
    }


def test_coerce_transcript_segments_normalizes_iterables():
    segments = (
        {"start": "0", "end": "1", "text": " one ", "speaker": "Alice"},
        {"start": None, "text": "two"},
    )

    assert coerce_transcript_segments(segments) == [
        {"start": 0.0, "end": 1.0, "text": "one", "speaker": "Alice"},
        {"start": 0.0, "end": 0.0, "text": "two"},
    ]
