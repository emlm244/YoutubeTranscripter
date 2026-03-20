from __future__ import annotations

from pathlib import Path

from config import AppConfig
import launcher_preflight as launcher_preflight


def test_inspect_whisper_model_reports_cached(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "whisper-model"
    model_dir.mkdir()

    monkeypatch.setattr(
        launcher_preflight,
        "resolve_whisper_model_from_cache",
        lambda model_name: model_dir,
    )

    item = launcher_preflight.inspect_whisper_model("large-v3")

    assert item.status == "ok"
    assert item.label == "Whisper model 'large-v3'"
    assert str(model_dir) in item.detail


def test_inspect_whisper_model_reports_missing_cache(monkeypatch):
    def _raise_missing(model_name: str):
        raise FileNotFoundError(model_name)

    monkeypatch.setattr(
        launcher_preflight,
        "resolve_whisper_model_from_cache",
        _raise_missing,
    )

    item = launcher_preflight.inspect_whisper_model("large-v3")

    assert item.status == "info"
    assert "First use will download" in item.detail


def test_inspect_gector_model_reports_cached(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "pytorch_model.bin"
    config_path.write_text("{}", encoding="utf-8")
    weights_path.write_bytes(b"weights")

    def _resolve(repo_id: str, filename: str) -> Path:
        return {
            "config.json": config_path,
            "pytorch_model.bin": weights_path,
        }[filename]

    monkeypatch.setattr(
        launcher_preflight,
        "resolve_hf_file_from_cache",
        _resolve,
    )

    item = launcher_preflight.inspect_gector_model("example/gector")

    assert item.status == "ok"
    assert item.label == "Grammar model 'example/gector'"
    assert str(tmp_path) in item.detail


def test_collect_preflight_items_deduplicates_matching_whisper_models(monkeypatch):
    monkeypatch.setattr(
        launcher_preflight,
        "inspect_whisper_model",
        lambda model_name: launcher_preflight.PreflightItem(
            f"Whisper model '{model_name}'",
            "ok",
            "cached",
        ),
    )
    monkeypatch.setattr(
        launcher_preflight,
        "inspect_gector_model",
        lambda model_name: launcher_preflight.PreflightItem("Grammar", "ok", "cached"),
    )
    monkeypatch.setattr(
        launcher_preflight,
        "inspect_verb_dictionary",
        lambda: launcher_preflight.PreflightItem("Verb dict", "ok", "bundled"),
    )
    monkeypatch.setattr(
        launcher_preflight,
        "inspect_language_tool_runtime",
        lambda: launcher_preflight.PreflightItem("LanguageTool", "info", "warm-up"),
    )

    config = AppConfig()
    config.transcription.whisper_model = launcher_preflight.REALTIME_MODEL_NAME

    items = launcher_preflight.collect_preflight_items(config)
    whisper_items = [item for item in items if item.label.startswith("Whisper model")]

    assert len(whisper_items) == 1
    assert whisper_items[0].label == f"Whisper model '{launcher_preflight.REALTIME_MODEL_NAME}'"
