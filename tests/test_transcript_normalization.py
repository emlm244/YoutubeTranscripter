"""Regression tests for fidelity-oriented transcript normalization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from config import TranscriptionConfig
from transcript_types import make_transcript_segment
from youtube_transcriber import _maybe_preprocess_audio_path, _normalize_transcript_segments


def test_normalization_preserves_repeated_segments_by_default() -> None:
    """Faithful mode should keep exact repetitions unless the user opts into deduping."""
    segments = [
        make_transcript_segment(start=0.0, end=1.0, text="Repeat me."),
        make_transcript_segment(start=1.0, end=2.0, text="Repeat me."),
        make_transcript_segment(start=2.0, end=3.0, text="Repeat me."),
        make_transcript_segment(start=3.0, end=4.0, text="Repeat me."),
    ]

    transcript, normalized, removed_count, hallucination_count = _normalize_transcript_segments(
        segments,
        clean_fillers=False,
        filter_hallucinated=False,
        deduplicate_repetitions=False,
    )

    assert transcript == "Repeat me. Repeat me. Repeat me. Repeat me."
    assert [segment["text"] for segment in normalized] == ["Repeat me."] * 4
    assert removed_count == 0
    assert hallucination_count == 0


def test_normalization_filters_and_deduplicates_only_when_enabled() -> None:
    """Legacy cleanup behavior should still be available when explicitly enabled."""
    segments = [
        make_transcript_segment(start=0.0, end=1.0, text="Hello."),
        make_transcript_segment(start=1.0, end=2.0, text="Thank you for watching!"),
        make_transcript_segment(start=2.0, end=3.0, text="Loop."),
        make_transcript_segment(start=3.0, end=4.0, text="Loop."),
        make_transcript_segment(start=4.0, end=5.0, text="Loop."),
        make_transcript_segment(start=5.0, end=6.0, text="Loop."),
    ]

    transcript, normalized, removed_count, hallucination_count = _normalize_transcript_segments(
        segments,
        clean_fillers=False,
        filter_hallucinated=True,
        deduplicate_repetitions=True,
    )

    assert transcript == "Hello. Loop."
    assert [segment["text"] for segment in normalized] == ["Hello.", "Loop."]
    assert removed_count == 3
    assert hallucination_count == 1


def test_preprocess_audio_path_uses_shared_file_pipeline(monkeypatch, tmp_path: Path) -> None:
    """Local-file preprocessing should route through the shared helper."""
    input_path = tmp_path / "input.wav"
    input_path.write_text("placeholder", encoding="utf-8")
    preprocess_calls: dict[str, object] = {}

    def _fake_preprocess_file(source: str, output: str, **kwargs):
        Path(output).write_text("processed", encoding="utf-8")
        preprocess_calls["source"] = source
        preprocess_calls["output"] = output
        preprocess_calls["kwargs"] = kwargs
        return output

    monkeypatch.setattr("youtube_transcriber.preprocess_file", _fake_preprocess_file)
    monkeypatch.setattr("youtube_transcriber._ffmpeg_executable", lambda name, ffmpeg_location: "ffmpeg-test")

    temp_paths: list[str] = []
    result = _maybe_preprocess_audio_path(
        str(input_path),
        ffmpeg_location="C:\\ffmpeg\\bin",
        config=TranscriptionConfig(noise_reduction_enabled=False, normalize_audio=True),
        temp_paths=temp_paths,
    )

    assert preprocess_calls["source"] == str(input_path)
    assert preprocess_calls["output"] == result
    assert preprocess_calls["kwargs"] == {
        "noise_reduction": False,
        "normalize": True,
        "ffmpeg_cmd": "ffmpeg-test",
    }
    assert temp_paths == [result]
    Path(result).unlink(missing_ok=True)


def test_preprocess_audio_path_decodes_non_wav_before_preprocessing(monkeypatch, tmp_path: Path) -> None:
    """Noise reduction should decode non-WAV inputs before passing them to the WAV-based preprocessor."""
    input_path = tmp_path / "input.mp3"
    input_path.write_text("placeholder", encoding="utf-8")
    decode_cmd: list[str] = []
    preprocess_source = {"path": ""}
    preprocess_output = {"path": ""}

    def _fake_run(cmd, check, capture_output, timeout):
        decode_cmd[:] = list(cmd)
        assert timeout > 0
        Path(cmd[-1]).write_text("decoded", encoding="utf-8")
        return SimpleNamespace()

    def _fake_preprocess_file(source: str, output: str, **kwargs):
        preprocess_source["path"] = source
        preprocess_output["path"] = output
        Path(output).write_text("processed", encoding="utf-8")
        return output

    monkeypatch.setattr("youtube_transcriber.subprocess.run", _fake_run)
    monkeypatch.setattr("youtube_transcriber.preprocess_file", _fake_preprocess_file)
    monkeypatch.setattr("youtube_transcriber._ffmpeg_executable", lambda name, ffmpeg_location: "ffmpeg-test")

    temp_paths: list[str] = []
    result = _maybe_preprocess_audio_path(
        str(input_path),
        ffmpeg_location="C:\\ffmpeg\\bin",
        config=TranscriptionConfig(noise_reduction_enabled=True, normalize_audio=False),
        temp_paths=temp_paths,
    )

    assert decode_cmd[0] == "ffmpeg-test"
    assert preprocess_source["path"].endswith(".wav")
    assert preprocess_source["path"] != str(input_path)
    assert result == preprocess_output["path"]
    assert temp_paths == [result]
    assert not Path(preprocess_source["path"]).exists()
    Path(result).unlink(missing_ok=True)


def test_normalization_strips_prompt_leakage_without_dropping_real_speech() -> None:
    """Decoder instruction leakage should be removed even when surrounding speech is real."""
    segments = [
        make_transcript_segment(
            start=0.0,
            end=2.0,
            text=(
                "Premium is the high class option. Do not paraphrase or rewrite for grammar. "
                "Use punctuation and capitalization only to make the spoken words readable. "
                "The wizard will also have rules."
            ),
        )
    ]

    transcript, normalized, removed_count, hallucination_count = _normalize_transcript_segments(
        segments,
        clean_fillers=False,
        filter_hallucinated=True,
        deduplicate_repetitions=True,
    )

    assert "do not paraphrase" not in transcript.lower()
    assert transcript == "Premium is the high class option. The wizard will also have rules."
    assert [segment["text"] for segment in normalized] == [
        "Premium is the high class option. The wizard will also have rules."
    ]
    assert removed_count == 0
    assert hallucination_count == 0
