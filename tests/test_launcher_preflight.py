from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    backbone_dir = tmp_path / "backbone"
    config_path.write_text(json.dumps({"model_id": "roberta-large"}), encoding="utf-8")
    weights_path.write_bytes(b"weights")
    tokenizer_config_path.write_text("{}", encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")

    def _resolve(repo_id: str, filename: str) -> Path:
        paths = {
            "config.json": config_path,
            "pytorch_model.bin": weights_path,
            "tokenizer_config.json": tokenizer_config_path,
            "tokenizer.json": tokenizer_path,
        }
        try:
            return paths[filename]
        except KeyError as exc:
            raise FileNotFoundError(filename) from exc

    monkeypatch.setattr(
        launcher_preflight,
        "resolve_hf_file_from_cache",
        _resolve,
    )
    monkeypatch.setattr(
        launcher_preflight,
        "_resolve_transformer_model_from_cache",
        lambda repo_id: backbone_dir,
    )

    item = launcher_preflight.inspect_gector_model("example/gector")

    assert item.status == "ok"
    assert item.label == "Grammar model 'example/gector'"
    assert str(tmp_path) in item.detail
    assert str(backbone_dir) in item.detail


def test_inspect_gector_model_accepts_safetensors_only_cache(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "model.safetensors"
    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    config_path.write_text("{}", encoding="utf-8")
    weights_path.write_bytes(b"weights")
    tokenizer_config_path.write_text("{}", encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")

    def _resolve(repo_id: str, filename: str) -> Path:
        paths = {
            "config.json": config_path,
            "model.safetensors": weights_path,
            "tokenizer_config.json": tokenizer_config_path,
            "tokenizer.json": tokenizer_path,
        }
        try:
            return paths[filename]
        except KeyError as exc:
            raise FileNotFoundError(filename) from exc

    monkeypatch.setattr(launcher_preflight, "resolve_hf_file_from_cache", _resolve)

    item = launcher_preflight.inspect_gector_model("example/gector")

    assert item.status == "ok"
    assert str(tmp_path) in item.detail


def test_resolve_transformer_model_from_cache_rejects_mixed_snapshots(monkeypatch, tmp_path: Path):
    model_snapshot = tmp_path / "model"
    tokenizer_snapshot = tmp_path / "tokenizer"
    model_snapshot.mkdir()
    tokenizer_snapshot.mkdir()
    paths = {
        "config.json": model_snapshot / "config.json",
        "model.safetensors": model_snapshot / "model.safetensors",
        "tokenizer_config.json": tokenizer_snapshot / "tokenizer_config.json",
        "tokenizer.json": tokenizer_snapshot / "tokenizer.json",
    }
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")

    def _resolve(repo_id: str, filename: str) -> Path:
        try:
            return paths[filename]
        except KeyError as exc:
            raise FileNotFoundError(filename) from exc

    monkeypatch.setattr(launcher_preflight, "resolve_hf_file_from_cache", _resolve)

    with pytest.raises(FileNotFoundError, match="multiple snapshots"):
        launcher_preflight._resolve_transformer_model_from_cache("example/backbone")


def test_bootstrap_runtime_environment_delegates_to_runtime_bootstrap(monkeypatch):
    calls = {"count": 0}

    def _bootstrap_runtime():
        calls["count"] += 1

    monkeypatch.setitem(
        sys.modules,
        "runtime_bootstrap",
        SimpleNamespace(bootstrap_runtime=_bootstrap_runtime),
    )

    launcher_preflight._bootstrap_runtime_environment()

    assert calls["count"] == 1


def test_resolve_whisper_model_from_cache_uses_bootstrap_and_cache_lookup(monkeypatch, tmp_path: Path):
    bootstrap_calls = {"count": 0}
    snapshot_dir = tmp_path / "hub" / "models--example--whisper" / "snapshots" / "abc123"
    cached_files = {
        "config.json": snapshot_dir / "config.json",
        "model.bin": snapshot_dir / "model.bin",
        "tokenizer.json": snapshot_dir / "tokenizer.json",
        "preprocessor_config.json": snapshot_dir / "preprocessor_config.json",
        "vocabulary.json": snapshot_dir / "vocabulary.json",
    }
    for path in cached_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")

    def _bootstrap_runtime():
        bootstrap_calls["count"] += 1

    def _try_to_load_from_cache(repo_id: str, filename: str):
        return str(cached_files[filename])

    monkeypatch.setattr(
        launcher_preflight,
        "_bootstrap_runtime_environment",
        _bootstrap_runtime,
    )
    monkeypatch.setattr(
        launcher_preflight,
        "FASTER_WHISPER_MODEL_REPOS",
        {"large-v3": "example/whisper"},
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=_try_to_load_from_cache),
    )

    resolved = launcher_preflight.resolve_whisper_model_from_cache("large-v3")

    assert bootstrap_calls["count"] == 1
    assert resolved == snapshot_dir.resolve()


def test_resolve_whisper_model_from_cache_rejects_partial_cache(monkeypatch, tmp_path: Path):
    snapshot_dir = tmp_path / "hub" / "models--example--whisper" / "snapshots" / "abc123"
    cached_files = {
        "config.json": snapshot_dir / "config.json",
        "model.bin": snapshot_dir / "model.bin",
        "tokenizer.json": snapshot_dir / "tokenizer.json",
        "preprocessor_config.json": snapshot_dir / "preprocessor_config.json",
    }
    for path in cached_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")

    monkeypatch.setattr(launcher_preflight, "_bootstrap_runtime_environment", lambda: None)
    monkeypatch.setattr(
        launcher_preflight,
        "FASTER_WHISPER_MODEL_REPOS",
        {"large-v3": "example/whisper"},
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            try_to_load_from_cache=(
                lambda repo_id, filename: str(cached_files[filename])
                if filename in cached_files
                else None
            )
        ),
    )

    with pytest.raises(FileNotFoundError, match="vocab"):
        launcher_preflight.resolve_whisper_model_from_cache("large-v3")


def test_resolve_hf_file_from_cache_uses_bootstrap_and_requires_cached_file(monkeypatch, tmp_path: Path):
    bootstrap_calls = {"count": 0}
    cached_file = tmp_path / "hub" / "models--example--gector" / "snapshots" / "abc123" / "config.json"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_text("{}", encoding="utf-8")

    def _bootstrap_runtime():
        bootstrap_calls["count"] += 1

    monkeypatch.setattr(
        launcher_preflight,
        "_bootstrap_runtime_environment",
        _bootstrap_runtime,
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

    with pytest.raises(FileNotFoundError, match="config\\.json"):
        launcher_preflight.resolve_hf_file_from_cache("example/gector", "config.json")


def test_resolve_whisper_model_from_cache_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr(launcher_preflight, "_bootstrap_runtime_environment", lambda: None)
    monkeypatch.setattr(launcher_preflight, "FASTER_WHISPER_MODEL_REPOS", {})

    with pytest.raises(ValueError, match="Unknown Whisper model"):
        launcher_preflight.resolve_whisper_model_from_cache("unknown-model")


def test_inspect_language_tool_runtime_reports_missing_assets_as_info(monkeypatch):
    monkeypatch.setattr(
        launcher_preflight,
        "get_languagetool_runtime_status",
        lambda: LanguageToolRuntimeStatus(False, "not cached", state="missing_assets"),
    )

    item = launcher_preflight.inspect_language_tool_runtime()

    assert item.status == "info"
    assert "First fallback use will download" in item.detail


def test_inspect_gector_model_reports_partial_cache_as_info(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    config_path.write_text("{}", encoding="utf-8")
    tokenizer_config_path.write_text("{}", encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")

    def _resolve(repo_id: str, filename: str) -> Path:
        paths = {
            "config.json": config_path,
            "tokenizer_config.json": tokenizer_config_path,
            "tokenizer.json": tokenizer_path,
        }
        return paths.get(filename, tmp_path / "missing.bin")

    monkeypatch.setattr(launcher_preflight, "resolve_hf_file_from_cache", _resolve)

    item = launcher_preflight.inspect_gector_model("example/gector")

    assert item.status == "info"
    assert "Some cached files are missing" in item.detail


def test_inspect_gector_model_reports_mixed_snapshot_cache_as_info(monkeypatch, tmp_path: Path):
    model_snapshot = tmp_path / "model"
    tokenizer_snapshot = tmp_path / "tokenizer"
    model_snapshot.mkdir()
    tokenizer_snapshot.mkdir()
    paths = {
        "config.json": model_snapshot / "config.json",
        "pytorch_model.bin": model_snapshot / "pytorch_model.bin",
        "tokenizer_config.json": tokenizer_snapshot / "tokenizer_config.json",
        "tokenizer.json": tokenizer_snapshot / "tokenizer.json",
    }
    paths["config.json"].write_text("{}", encoding="utf-8")
    for filename, path in paths.items():
        if filename != "config.json":
            path.write_text("{}", encoding="utf-8")

    def _resolve(repo_id: str, filename: str) -> Path:
        try:
            return paths[filename]
        except KeyError as exc:
            raise FileNotFoundError(filename) from exc

    monkeypatch.setattr(launcher_preflight, "resolve_hf_file_from_cache", _resolve)

    item = launcher_preflight.inspect_gector_model("example/gector")

    assert item.status == "info"
    assert "Not cached yet" in item.detail


def test_inspect_gector_model_reports_missing_backbone_as_info(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "pytorch_model.bin"
    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    config_path.write_text(json.dumps({"model_id": "roberta-large"}), encoding="utf-8")
    weights_path.write_bytes(b"weights")
    tokenizer_config_path.write_text("{}", encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")

    def _resolve(repo_id: str, filename: str) -> Path:
        paths = {
            "config.json": config_path,
            "pytorch_model.bin": weights_path,
            "tokenizer_config.json": tokenizer_config_path,
            "tokenizer.json": tokenizer_path,
        }
        try:
            return paths[filename]
        except KeyError as exc:
            raise FileNotFoundError(filename) from exc

    monkeypatch.setattr(launcher_preflight, "resolve_hf_file_from_cache", _resolve)

    def _raise_file_not_found(repo_id: str) -> Path:
        raise FileNotFoundError(repo_id)

    monkeypatch.setattr(
        launcher_preflight,
        "_resolve_transformer_model_from_cache",
        _raise_file_not_found,
    )

    item = launcher_preflight.inspect_gector_model("example/gector")

    assert item.status == "info"
    assert "backbone 'roberta-large' is not cached yet" in item.detail


def test_inspect_verb_dictionary_reports_presence(monkeypatch, tmp_path: Path):
    verb_dict = tmp_path / "verb-form-vocab.txt"
    verb_dict.write_text("be was been", encoding="utf-8")
    monkeypatch.setattr(launcher_preflight, "get_verb_dictionary_path", lambda: verb_dict)

    item = launcher_preflight.inspect_verb_dictionary()

    assert item.status == "ok"
    assert str(verb_dict) in item.detail


def test_inspect_verb_dictionary_reports_missing(monkeypatch):
    monkeypatch.setattr(launcher_preflight, "get_verb_dictionary_path", lambda: None)

    item = launcher_preflight.inspect_verb_dictionary()

    assert item.status == "info"
    assert "Not bundled yet" in item.detail


def test_inspect_language_tool_runtime_reports_available(monkeypatch):
    monkeypatch.setattr(
        launcher_preflight,
        "get_languagetool_runtime_status",
        lambda: LanguageToolRuntimeStatus(True, "Ready", state="ready"),
    )

    item = launcher_preflight.inspect_language_tool_runtime()

    assert item.status == "ok"
    assert item.detail == "Ready"


def test_inspect_language_tool_runtime_reports_other_failures_as_warning(monkeypatch):
    monkeypatch.setattr(
        launcher_preflight,
        "get_languagetool_runtime_status",
        lambda: LanguageToolRuntimeStatus(False, "Java missing", state="missing_java"),
    )

    item = launcher_preflight.inspect_language_tool_runtime()

    assert item.status == "warning"
    assert item.detail == "Java missing"


def test_collect_preflight_items_includes_single_supported_whisper_model(monkeypatch):
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
    config.transcription.batch_backend = "local_whisper"
    config.transcription.whisper_model = "large-v3-turbo"

    items = launcher_preflight.collect_preflight_items(config)
    whisper_items = [item for item in items if item.label.startswith("Whisper model")]

    assert [item.label for item in whisper_items] == [
        "Whisper model 'large-v3'",
    ]


def test_collect_preflight_items_skips_whisper_models_for_openai_backend(monkeypatch):
    inspected: list[str] = []

    def _inspect_whisper_model(model_name: str) -> launcher_preflight.PreflightItem:
        inspected.append(model_name)
        return launcher_preflight.PreflightItem(f"Whisper {model_name}", "ok", model_name)

    monkeypatch.setattr(
        launcher_preflight,
        "inspect_whisper_model",
        _inspect_whisper_model,
    )

    config = AppConfig()
    config.transcription.batch_backend = "openai"

    items = launcher_preflight.collect_preflight_items(config)

    assert items == []
    assert inspected == []


def test_collect_preflight_items_includes_grammar_checks_when_enabled(monkeypatch):
    monkeypatch.setattr(
        launcher_preflight,
        "inspect_whisper_model",
        lambda model_name: launcher_preflight.PreflightItem(f"Whisper {model_name}", "ok", model_name),
    )
    monkeypatch.setattr(
        launcher_preflight,
        "inspect_gector_model",
        lambda model_name: launcher_preflight.PreflightItem("Grammar", "ok", model_name),
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
    config.transcription.batch_backend = "local_whisper"
    config.grammar.enabled = True

    items = launcher_preflight.collect_preflight_items(config)

    assert [item.label for item in items] == [
        "Whisper large-v3",
        "Grammar",
        "Verb dict",
        "LanguageTool",
    ]


def test_format_item_and_main_output(monkeypatch, capsys):
    monkeypatch.setattr(
        launcher_preflight,
        "collect_preflight_items",
        lambda config=None: [launcher_preflight.PreflightItem("Whisper", "ok", "cached")],
    )

    item = launcher_preflight.PreflightItem("Whisper", "warning", "missing")
    assert launcher_preflight.format_item(item) == "[WARNING] Whisper: missing"

    assert launcher_preflight.main() == 0
    output = capsys.readouterr().out
    assert "[INFO] Runtime asset preflight:" in output
    assert "[OK] Whisper: cached" in output
