from __future__ import annotations

import json
from pathlib import Path

import build_cache_manifest as bcm
from config import AppConfig


def test_repo_root_returns_model_cache_root(tmp_path: Path):
    repo = tmp_path / "hf-cache" / "hub" / "models--openai--whisper-large-v3"
    snapshot = repo / "snapshots" / "abc123" / "model.bin"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("x", encoding="utf-8")

    assert bcm.repo_root(snapshot) == str(repo.resolve())


def test_main_emits_unique_cached_repo_roots(monkeypatch, capsys, tmp_path: Path):
    config = AppConfig()
    config.transcription.whisper_model = "large-v3"
    config.grammar.gector_model = "example/gector"

    whisper_repo = tmp_path / "hub" / "models--openai--whisper-large-v3"
    realtime_repo = tmp_path / "hub" / "models--openai--distil-large-v3"
    grammar_repo = tmp_path / "hub" / "models--example--gector"
    whisper_snapshot = whisper_repo / "snapshots" / "a" / "model.bin"
    realtime_snapshot = realtime_repo / "snapshots" / "b" / "model.bin"
    grammar_config = grammar_repo / "snapshots" / "c" / "config.json"
    grammar_weights = grammar_repo / "snapshots" / "c" / "pytorch_model.bin"
    for path in (whisper_snapshot, realtime_snapshot, grammar_config, grammar_weights):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(bcm, "get_config", lambda: config)
    monkeypatch.setattr(
        bcm,
        "resolve_whisper_model_from_cache",
        lambda model_name: {
            "large-v3": whisper_snapshot,
            bcm.REALTIME_MODEL_NAME: realtime_snapshot,
        }[model_name],
    )
    monkeypatch.setattr(
        bcm,
        "resolve_hf_file_from_cache",
        lambda repo_id, filename: {
            "config.json": grammar_config,
            "pytorch_model.bin": grammar_weights,
        }[filename],
    )

    assert bcm.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == [
        str(whisper_repo.resolve()),
        str(realtime_repo.resolve()),
        str(grammar_repo.resolve()),
    ]


def test_main_skips_duplicate_realtime_model(monkeypatch, capsys, tmp_path: Path):
    config = AppConfig()
    config.transcription.whisper_model = bcm.REALTIME_MODEL_NAME
    config.grammar.enabled = False

    realtime_repo = tmp_path / "hub" / "models--openai--distil-large-v3"
    realtime_snapshot = realtime_repo / "snapshots" / "b" / "model.bin"
    realtime_snapshot.parent.mkdir(parents=True, exist_ok=True)
    realtime_snapshot.write_text("x", encoding="utf-8")

    calls: list[str] = []

    monkeypatch.setattr(bcm, "get_config", lambda: config)
    monkeypatch.setattr(
        bcm,
        "resolve_whisper_model_from_cache",
        lambda model_name: calls.append(model_name) or realtime_snapshot,
    )

    assert bcm.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert calls == [bcm.REALTIME_MODEL_NAME]
    assert payload == [str(realtime_repo.resolve())]
