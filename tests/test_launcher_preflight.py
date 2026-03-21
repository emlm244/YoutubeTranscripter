from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from config import AppConfig
from grammar_postprocessor import LanguageToolRuntimeStatus
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


def test_resolve_whisper_model_from_cache_uses_bootstrap_and_cache_lookup(monkeypatch, tmp_path: Path):
    bootstrap_calls = {"count": 0}
    snapshot_dir = tmp_path / "hub" / "models--example--whisper" / "snapshots" / "abc123"
    config_path = snapshot_dir / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        launcher_preflight,
        "_bootstrap_runtime_environment",
        lambda: bootstrap_calls.__setitem__("count", bootstrap_calls["count"] + 1),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", SimpleNamespace(_MODELS={"large-v3": "example/whisper"}))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=lambda repo_id, filename: str(config_path)),
    )

    resolved = launcher_preflight.resolve_whisper_model_from_cache("large-v3")

    assert bootstrap_calls["count"] == 1
    assert resolved == snapshot_dir.resolve()


def test_resolve_hf_file_from_cache_uses_bootstrap_and_requires_cached_file(monkeypatch, tmp_path: Path):
    bootstrap_calls = {"count": 0}
    cached_file = tmp_path / "hub" / "models--example--gector" / "snapshots" / "abc123" / "config.json"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        launcher_preflight,
        "_bootstrap_runtime_environment",
        lambda: bootstrap_calls.__setitem__("count", bootstrap_calls["count"] + 1),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=lambda repo_id, filename: str(cached_file)),
    )

    resolved = launcher_preflight.resolve_hf_file_from_cache("example/gector", "config.json")

    assert bootstrap_calls["count"] == 1
    assert resolved == cached_file.resolve()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=lambda repo_id, filename: None),
    )

    try:
        launcher_preflight.resolve_hf_file_from_cache("example/gector", "config.json")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError when cache lookup misses")


def test_inspect_language_tool_runtime_reports_missing_assets_as_info(monkeypatch):
    monkeypatch.setattr(
        launcher_preflight,
        "get_languagetool_runtime_status",
        lambda: LanguageToolRuntimeStatus(False, "not cached", state="missing_assets"),
    )

    item = launcher_preflight.inspect_language_tool_runtime()

    assert item.status == "info"
    assert "First fallback use will download" in item.detail


def test_inspect_language_tool_runtime_reports_other_failures_as_warning(monkeypatch):
    monkeypatch.setattr(
        launcher_preflight,
        "get_languagetool_runtime_status",
        lambda: LanguageToolRuntimeStatus(False, "Java missing", state="missing_java"),
    )

    item = launcher_preflight.inspect_language_tool_runtime()

    assert item.status == "warning"
    assert item.detail == "Java missing"


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
