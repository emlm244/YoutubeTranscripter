"""Shared typed transcript segment definitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import NotRequired, TypeAlias, TypedDict


class TranscriptSegment(TypedDict):
    """Normalized transcript segment structure shared across modules."""

    start: float
    end: float
    text: str
    speaker: NotRequired[str]


TranscriptSegments = list[TranscriptSegment]
TranscriptSegmentLike: TypeAlias = Mapping[str, object]


def _coerce_float(value: object) -> float:
    """Best-effort float coercion for legacy segment payloads."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def make_transcript_segment(
    *,
    start: float,
    end: float,
    text: str,
    speaker: str | None = None,
) -> TranscriptSegment:
    """Build a normalized transcript segment."""
    segment: TranscriptSegment = {
        "start": float(start),
        "end": float(end),
        "text": text.strip(),
    }
    if speaker:
        segment["speaker"] = speaker
    return segment


def replace_segment_text(segment: TranscriptSegment, text: str) -> TranscriptSegment:
    """Return a copied segment with updated text."""
    updated: TranscriptSegment = {
        "start": segment.get("start", 0.0),
        "end": segment.get("end", 0.0),
        "text": text,
    }
    speaker = segment.get("speaker")
    if speaker:
        updated["speaker"] = speaker
    return updated


def coerce_transcript_segment(segment: TranscriptSegmentLike) -> TranscriptSegment:
    """Normalize a dict-like segment into the shared typed structure."""
    speaker = segment.get("speaker")
    return make_transcript_segment(
        start=_coerce_float(segment.get("start", 0.0)),
        end=_coerce_float(segment.get("end", 0.0)),
        text=str(segment.get("text", "")),
        speaker=speaker if isinstance(speaker, str) and speaker.strip() else None,
    )


def coerce_transcript_segments(segments: Iterable[TranscriptSegmentLike]) -> TranscriptSegments:
    """Normalize an iterable of dict-like segments."""
    return [coerce_transcript_segment(segment) for segment in segments]
