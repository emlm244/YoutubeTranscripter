"""Tests for config.py - TranscriptionConfig, presets, AppConfig loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import config
from config import (
    ACCURACY_PRESETS,
    DEFAULT_PRESET,
    AppConfig,
    GrammarConfig,
    TranscriptionConfig,
    apply_preset,
)


class TestTranscriptionConfig:
    """Test TranscriptionConfig dataclass defaults and fields."""

    def test_defaults(self):
        cfg = TranscriptionConfig()
        assert cfg.whisper_model == "large-v3"
        assert cfg.beam_size == 5
        assert cfg.temperature == 0.0
        assert cfg.vad_filter is True
        assert cfg.word_timestamps is True
        assert cfg.initial_prompt is None
        assert cfg.language is None
        assert cfg.hotwords is None
        assert cfg.condition_on_previous_text is True
        assert cfg.clean_filler_words is True
        assert cfg.noise_reduction_enabled is True
        assert cfg.normalize_audio is True

    def test_custom_values(self):
        cfg = TranscriptionConfig(
            whisper_model="distil-large-v3",
            language="en",
            hotwords="Anthropic, Claude",
            clean_filler_words=False,
        )
        assert cfg.whisper_model == "distil-large-v3"
        assert cfg.language == "en"
        assert cfg.hotwords == "Anthropic, Claude"
        assert cfg.clean_filler_words is False

    def test_vad_defaults_for_accented_speech(self):
        cfg = TranscriptionConfig()
        assert cfg.vad_threshold == 0.25
        assert cfg.min_speech_duration_ms == 50
        assert cfg.min_silence_duration_ms == 2000
        assert cfg.speech_pad_ms == 400


class TestAccuracyPresets:
    """Test accuracy presets system."""

    def test_all_presets_exist(self):
        assert "speed" in ACCURACY_PRESETS
        assert "balanced" in ACCURACY_PRESETS
        assert "max_accuracy" in ACCURACY_PRESETS

    def test_default_preset_is_max_accuracy(self):
        assert DEFAULT_PRESET == "max_accuracy"

    def test_speed_preset_uses_distil(self):
        preset = ACCURACY_PRESETS["speed"]
        assert preset.whisper_model == "distil-large-v3"
        assert preset.language is None

    def test_max_accuracy_preset_uses_large_v3(self):
        preset = ACCURACY_PRESETS["max_accuracy"]
        assert preset.whisper_model == "large-v3"
        assert preset.language == "en"
        assert preset.initial_prompt is not None
        assert "meeting" in preset.initial_prompt.lower()

    def test_apply_preset_modifies_config(self):
        cfg = TranscriptionConfig()
        apply_preset(cfg, "speed")
        assert cfg.whisper_model == "distil-large-v3"
        assert cfg.language is None
        assert cfg.vad_threshold == 0.35

    def test_apply_invalid_preset_noop(self):
        cfg = TranscriptionConfig()
        original_model = cfg.whisper_model
        apply_preset(cfg, "nonexistent")
        assert cfg.whisper_model == original_model

    def test_balanced_preset_values(self):
        preset = ACCURACY_PRESETS["balanced"]
        assert preset.whisper_model == "distil-large-v3"
        assert preset.language == "en"
        assert preset.vad_threshold == 0.30


class TestGrammarConfig:
    """Test GrammarConfig dataclass."""

    def test_defaults(self):
        cfg = GrammarConfig()
        assert cfg.enabled is True
        assert cfg.backend == "auto"

    def test_custom_backend(self):
        cfg = GrammarConfig(backend="gector")
        assert cfg.backend == "gector"


class TestAppConfig:
    """Test AppConfig load and save."""

    def test_default_config(self):
        cfg = AppConfig()
        assert cfg.gpu_memory_fraction > 0
        assert cfg.transcription is not None
        assert cfg.grammar is not None

    def test_load_from_json(self):
        data = {
            "gpu_memory_fraction": 0.80,
            "transcription": {
                "whisper_model": "distil-large-v3",
                "language": "en",
                "hotwords": "test",
                "clean_filler_words": False,
                "noise_reduction_enabled": False,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.gpu_memory_fraction == 0.80
        assert cfg.transcription.whisper_model == "distil-large-v3"
        assert cfg.transcription.language == "en"
        assert cfg.transcription.hotwords == "test"
        assert cfg.transcription.clean_filler_words is False
        assert cfg.transcription.noise_reduction_enabled is False

    def test_load_missing_file_returns_defaults(self):
        cfg = AppConfig.load(Path("/nonexistent/path/config.json"))
        assert cfg.transcription.whisper_model == "large-v3"

    def test_load_invalid_system_values_fallback_to_defaults(self):
        data = {
            "gpu_memory_fraction": 1.5,
            "max_audio_size_mb": 0,
            "max_filename_length": -10,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.gpu_memory_fraction == 0.90
        assert cfg.max_audio_size_mb == 500
        assert cfg.max_filename_length == 200

    def test_load_rejects_bool_system_values(self):
        data = {
            "gpu_memory_fraction": True,
            "max_audio_size_mb": False,
            "max_filename_length": True,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.gpu_memory_fraction == 0.90
        assert cfg.max_audio_size_mb == 500
        assert cfg.max_filename_length == 200

    def test_load_valid_system_values(self):
        data = {
            "gpu_memory_fraction": 0.75,
            "max_audio_size_mb": 750,
            "max_filename_length": 180,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.gpu_memory_fraction == 0.75
        assert cfg.max_audio_size_mb == 750
        assert cfg.max_filename_length == 180

    def test_post_init_validates_system_limits(self):
        with pytest.raises(ValueError):
            AppConfig(gpu_memory_fraction=0)
        with pytest.raises(ValueError):
            AppConfig(max_audio_size_mb=0)
        with pytest.raises(ValueError):
            AppConfig(max_filename_length=0)

    def test_save_and_reload_roundtrip_including_ui_and_grammar(self):
        cfg = AppConfig()
        cfg.ui.output_format = "timestamped"
        cfg.grammar.backend = "languagetool"
        cfg.max_audio_size_mb = 650

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        cfg.save(path)
        loaded = AppConfig.load(path)

        assert loaded.ui.output_format == "timestamped"
        assert loaded.grammar.backend == "languagetool"
        assert loaded.max_audio_size_mb == 650

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert set(raw["ui"]) == {"last_youtube_url", "output_format", "transcription_preset"}

    def test_load_ignores_legacy_ui_fields(self):
        data = {
            "ui": {
                "theme": "dark",
                "accent_color": "blue",
                "window_width": 1280,
                "window_height": 1024,
                "splitter_ratios": [0.4, 0.2, 0.4],
                "remember_window_position": True,
                "last_youtube_url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
                "output_format": "timestamped",
                "transcription_preset": "balanced",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.ui.last_youtube_url.endswith("dQw4w9WgXcQ")
        assert cfg.ui.output_format == "timestamped"
        assert cfg.ui.transcription_preset == "balanced"

    def test_load_invalid_json_returns_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not-json")
            f.flush()
            cfg = AppConfig.load(Path(f.name))
        assert cfg.transcription.whisper_model == "large-v3"

    def test_global_get_config_is_singleton_and_save_config_writes(self, monkeypatch):
        monkeypatch.setattr(config, "_config", None)
        saved = {"called": 0}

        first = config.get_config()
        second = config.get_config()
        assert first is second

        def fake_save(self, _path=None):
            saved["called"] += 1

        monkeypatch.setattr(type(first), "save", fake_save)
        config.save_config()
        assert saved["called"] == 1

    def test_save_without_path_uses_helper_default(self, monkeypatch, tmp_path: Path):
        target = tmp_path / "config.json"
        monkeypatch.setattr(config, "get_config_path", lambda: target)

        cfg = AppConfig()
        cfg.transcription.whisper_model = "distil-large-v3"
        cfg.save()

        raw = json.loads(target.read_text(encoding="utf-8"))
        assert raw["transcription"]["whisper_model"] == "distil-large-v3"

    def test_load_without_path_uses_helper_default(self, monkeypatch, tmp_path: Path):
        target = tmp_path / "config.json"
        target.write_text(
            json.dumps({"transcription": {"whisper_model": "distil-large-v3"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "get_config_path", lambda: target)

        loaded = AppConfig.load()

        assert loaded.transcription.whisper_model == "distil-large-v3"
