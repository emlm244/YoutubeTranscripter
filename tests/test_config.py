"""Tests for config.py - TranscriptionConfig, presets, AppConfig loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import config
from config import (
    ACCURACY_PRESETS,
    BATCH_BACKEND_OPTIONS,
    COMPUTE_TYPE_OPTIONS,
    DEVICE_PREFERENCE_OPTIONS,
    DEFAULT_PRESET,
    GRAMMAR_BACKEND_OPTIONS,
    AppConfig,
    GrammarConfig,
    OPENAI_BATCH_MODEL_OPTIONS,
    RecordingConfig,
    TranscriptionConfig,
    get_whisper_models_for_runtime,
    normalize_whisper_model_name,
    apply_preset,
)


class TestTranscriptionConfig:
    """Test TranscriptionConfig dataclass defaults and fields."""

    def test_defaults(self):
        cfg = TranscriptionConfig()
        assert cfg.batch_backend == "openai"
        assert cfg.openai_batch_model == "gpt-4o-transcribe"
        assert cfg.whisper_model == "large-v3"
        assert cfg.device_preference == "auto"
        assert cfg.compute_type == "auto"
        assert cfg.beam_size == 5
        assert cfg.temperature == 0.0
        assert cfg.vad_filter is True
        assert cfg.batch_size == 32
        assert cfg.cpu_fallback_batch_size == 8
        assert cfg.word_timestamps is True
        assert cfg.initial_prompt is None
        assert cfg.language is None
        assert cfg.hotwords is None
        assert cfg.condition_on_previous_text is False
        assert cfg.clean_filler_words is False
        assert cfg.filter_hallucinations is True
        assert cfg.deduplicate_repeated_segments is True
        assert cfg.noise_reduction_enabled is True
        assert cfg.normalize_audio is True

    def test_custom_values(self):
        cfg = TranscriptionConfig(
            batch_backend="compare",
            openai_batch_model="gpt-4o-mini-transcribe",
            whisper_model="large-v3",
            device_preference="cuda",
            compute_type="float16",
            language="en",
            hotwords="Anthropic, Claude",
            clean_filler_words=False,
        )
        assert cfg.batch_backend == "compare"
        assert cfg.openai_batch_model == "gpt-4o-mini-transcribe"
        assert cfg.whisper_model == "large-v3"
        assert cfg.device_preference == "cuda"
        assert cfg.compute_type == "float16"
        assert cfg.language == "en"
        assert cfg.hotwords == "Anthropic, Claude"
        assert cfg.clean_filler_words is False

    def test_vad_defaults_for_accented_speech(self):
        cfg = TranscriptionConfig()
        assert cfg.vad_threshold == 0.25
        assert cfg.min_speech_duration_ms == 50
        assert cfg.min_silence_duration_ms == 2000
        assert cfg.speech_pad_ms == 400


class TestRecordingConfig:
    """Test RecordingConfig dataclass defaults."""

    def test_defaults(self):
        cfg = RecordingConfig()

        assert cfg.default_microphone == ""
        assert cfg.sample_rate == 16000


class TestAccuracyPresets:
    """Test accuracy presets system."""

    def test_all_presets_exist(self):
        assert "speed" in ACCURACY_PRESETS
        assert "balanced" in ACCURACY_PRESETS
        assert "max_accuracy" in ACCURACY_PRESETS

    def test_default_preset_is_max_accuracy(self):
        assert DEFAULT_PRESET == "max_accuracy"

    def test_speed_preset_uses_large_v3_with_lighter_decoding(self):
        preset = ACCURACY_PRESETS["speed"]
        assert preset.whisper_model == "large-v3"
        assert preset.beam_size == 2
        assert preset.batch_size == 48
        assert preset.language is None
        assert preset.initial_prompt is None
        assert preset.condition_on_previous_text is False
        assert preset.filter_hallucinations is True
        assert preset.deduplicate_repeated_segments is True

    def test_max_accuracy_preset_uses_large_v3(self):
        preset = ACCURACY_PRESETS["max_accuracy"]
        assert preset.whisper_model == "large-v3"
        assert preset.beam_size == 5
        assert preset.batch_size == 32
        assert preset.language == "en"
        assert preset.initial_prompt is None
        assert preset.condition_on_previous_text is False

    def test_apply_preset_modifies_config(self):
        cfg = TranscriptionConfig()
        apply_preset(cfg, "speed")
        assert cfg.whisper_model == "large-v3"
        assert cfg.beam_size == 2
        assert cfg.batch_size == 48
        assert cfg.language is None
        assert cfg.vad_threshold == 0.35
        assert cfg.initial_prompt is None
        assert cfg.condition_on_previous_text is False
        assert cfg.filter_hallucinations is True
        assert cfg.deduplicate_repeated_segments is True

    def test_apply_invalid_preset_noop(self):
        cfg = TranscriptionConfig()
        original_model = cfg.whisper_model
        apply_preset(cfg, "nonexistent")
        assert cfg.whisper_model == original_model

    def test_balanced_preset_values(self):
        preset = ACCURACY_PRESETS["balanced"]
        assert preset.whisper_model == "large-v3"
        assert preset.beam_size == 4
        assert preset.batch_size == 40
        assert preset.language == "en"
        assert preset.vad_threshold == 0.30
        assert preset.initial_prompt is None

    def test_get_whisper_models_for_runtime_returns_single_supported_model(self):
        cfg = AppConfig()
        cfg.transcription.whisper_model = "large-v3"

        models = get_whisper_models_for_runtime(cfg)

        assert models == ["large-v3"]

    def test_normalize_whisper_model_name_maps_legacy_aliases_to_large_v3(self):
        assert normalize_whisper_model_name("distil-large-v3") == "large-v3"
        assert normalize_whisper_model_name("distil-large-v2") == "large-v3"
        assert normalize_whisper_model_name("large-v3-turbo") == "large-v3"
        assert normalize_whisper_model_name("turbo") == "large-v3"
        assert normalize_whisper_model_name("custom/repo") == "large-v3"


class TestGrammarConfig:
    """Test GrammarConfig dataclass."""

    def test_defaults(self):
        cfg = GrammarConfig()
        assert cfg.enabled is False
        assert cfg.backend == "auto"

    def test_custom_backend(self):
        cfg = GrammarConfig(backend="gector")
        assert cfg.backend == "gector"

    def test_backend_options_cover_defaults(self):
        assert ("auto", "Auto") in GRAMMAR_BACKEND_OPTIONS
        assert ("auto", "Auto") in DEVICE_PREFERENCE_OPTIONS
        assert ("auto", "Auto") in COMPUTE_TYPE_OPTIONS
        assert ("openai", "OpenAI") in BATCH_BACKEND_OPTIONS
        assert ("gpt-4o-transcribe", "GPT-4o Transcribe") in OPENAI_BATCH_MODEL_OPTIONS


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
                "batch_backend": "compare",
                "openai_batch_model": "gpt-4o-mini-transcribe",
                "whisper_model": "distil-large-v3",
                "device_preference": "cuda",
                "compute_type": "float16",
                "language": "en",
                "hotwords": "test",
                "clean_filler_words": False,
                "filter_hallucinations": True,
                "deduplicate_repeated_segments": True,
                "noise_reduction_enabled": False,
                "cpu_fallback_batch_size": 6,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.gpu_memory_fraction == 0.80
        assert cfg.transcription.batch_backend == "compare"
        assert cfg.transcription.openai_batch_model == "gpt-4o-mini-transcribe"
        assert cfg.transcription.whisper_model == "large-v3"
        assert cfg.transcription.device_preference == "cuda"
        assert cfg.transcription.compute_type == "float16"
        assert cfg.transcription.language == "en"
        assert cfg.transcription.hotwords == "test"
        assert cfg.transcription.cpu_fallback_batch_size == 6
        assert cfg.transcription.clean_filler_words is False
        assert cfg.transcription.filter_hallucinations is True
        assert cfg.transcription.deduplicate_repeated_segments is True
        assert cfg.transcription.noise_reduction_enabled is False

    def test_load_accepts_utf8_bom_json(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"transcription": {"whisper_model": "large-v3-turbo"}}),
            encoding="utf-8-sig",
        )

        cfg = AppConfig.load(path)

        assert cfg.transcription.whisper_model == "large-v3"

    def test_load_missing_file_returns_defaults(self):
        cfg = AppConfig.load(Path("/nonexistent/path/config.json"))
        assert cfg.transcription.batch_backend == "openai"
        assert cfg.transcription.whisper_model == "large-v3"
        assert cfg.transcription.initial_prompt is None
        assert cfg.recording.sample_rate == 16000
        assert cfg.grammar.enabled is False

    def test_load_ignores_legacy_realtime_recording_keys(self):
        data = {
            "recording": {
                "default_microphone": "Studio Mic",
                "sample_rate": 48000,
                "openai_realtime_session_model": "gpt-realtime-1.5",
                "openai_realtime_transcription_model": "gpt-4o-transcribe-latest",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.recording.default_microphone == "Studio Mic"
        assert cfg.recording.sample_rate == 48000
        assert not hasattr(cfg.recording, "openai_realtime_session_model")
        assert not hasattr(cfg.recording, "openai_realtime_transcription_model")

    def test_load_logs_legacy_model_migration_warning(self, caplog):
        data = {
            "transcription": {
                "whisper_model": "distil-large-v3",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            with caplog.at_level("WARNING", logger="config"):
                cfg = AppConfig.load(Path(f.name))

        assert cfg.transcription.whisper_model == "large-v3"
        assert "distil-large-v3" in caplog.text
        assert "migrated to 'large-v3'" in caplog.text

    def test_load_non_mapping_json_returns_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["not", "a", "mapping"], f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.transcription.whisper_model == "large-v3"
        assert cfg.transcription.batch_backend == "openai"
        assert cfg.recording.sample_rate == 16000
        assert cfg.ui.output_format == "plain"
        assert cfg.grammar.enabled is False

    def test_load_ignores_malformed_nested_sections(self):
        data = {
            "transcription": [],
            "recording": "bad-shape",
            "ui": 42,
            "grammar": ["bad-shape"],
            "gpu_memory_fraction": 0.75,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.transcription.whisper_model == "large-v3"
        assert cfg.transcription.batch_backend == "openai"
        assert cfg.recording.sample_rate == 16000
        assert cfg.ui.output_format == "plain"
        assert cfg.grammar.enabled is False
        assert cfg.gpu_memory_fraction == 0.75

    def test_load_disables_legacy_instruction_prompt(self):
        data = {
            "transcription": {
                "initial_prompt": (
                    "This is a faithful transcript of spoken audio. Preserve the speaker's exact wording, "
                    "filler words, repetitions, and unfinished thoughts. Do not paraphrase, summarize, or "
                    "rewrite for grammar. Use punctuation and capitalization only to reflect the spoken words."
                ),
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.transcription.initial_prompt is None
        assert cfg.transcription.condition_on_previous_text is False
        assert cfg.transcription.filter_hallucinations is True
        assert cfg.transcription.deduplicate_repeated_segments is True

    def test_load_migrates_legacy_prompt_flags_even_when_old_values_are_saved(self):
        data = {
            "transcription": {
                "initial_prompt": (
                    "This is a faithful transcript of spoken audio, including meetings with diverse accents. "
                    "Preserve exact wording, filler words, repetitions, and uncertain phrasing. "
                    "Do not paraphrase or rewrite for grammar. Use punctuation and capitalization only to "
                    "make the spoken words readable."
                ),
                "condition_on_previous_text": True,
                "filter_hallucinations": False,
                "deduplicate_repeated_segments": False,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.transcription.initial_prompt is None
        assert cfg.transcription.condition_on_previous_text is False
        assert cfg.transcription.filter_hallucinations is True
        assert cfg.transcription.deduplicate_repeated_segments is True

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

    def test_load_legacy_config_without_backend_uses_local_whisper(self):
        data = {"transcription": {"whisper_model": "large-v3"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cfg = AppConfig.load(Path(f.name))

        assert cfg.transcription.batch_backend == "local_whisper"

    def test_save_does_not_persist_openai_api_key(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        path = tmp_path / "config.json"
        cfg = AppConfig()
        cfg.save(path)

        raw_text = path.read_text(encoding="utf-8")
        assert "test-openai-key" not in raw_text
        assert "OPENAI_API_KEY" not in raw_text

    def test_post_init_validates_system_limits(self):
        with pytest.raises(ValueError, match="gpu_memory_fraction must be between 0 and 1"):
            AppConfig(gpu_memory_fraction=0)
        with pytest.raises(ValueError, match="max_audio_size_mb must be positive"):
            AppConfig(max_audio_size_mb=0)
        with pytest.raises(ValueError, match="max_filename_length must be positive"):
            AppConfig(max_filename_length=0)

    def test_save_and_reload_roundtrip_including_ui_and_grammar(self):
        cfg = AppConfig()
        cfg.ui.output_format = "timestamped"
        cfg.grammar.backend = "languagetool"
        cfg.max_audio_size_mb = 650
        cfg.transcription.device_preference = "cpu"
        cfg.transcription.compute_type = "int8"
        cfg.transcription.cpu_fallback_batch_size = 4

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        cfg.save(path)
        loaded = AppConfig.load(path)

        assert loaded.ui.output_format == "timestamped"
        assert loaded.grammar.backend == "languagetool"
        assert loaded.max_audio_size_mb == 650
        assert loaded.transcription.device_preference == "cpu"
        assert loaded.transcription.compute_type == "int8"
        assert loaded.transcription.cpu_fallback_batch_size == 4

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
        assert cfg.transcription.clean_filler_words is False

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
        cfg.transcription.whisper_model = "large-v3"
        cfg.save()

        raw = json.loads(target.read_text(encoding="utf-8"))
        assert raw["transcription"]["whisper_model"] == "large-v3"

    def test_load_without_path_uses_helper_default(self, monkeypatch, tmp_path: Path):
        target = tmp_path / "config.json"
        target.write_text(
            json.dumps({"transcription": {"whisper_model": "distil-large-v3"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "get_config_path", lambda: target)

        loaded = AppConfig.load()

        assert loaded.transcription.whisper_model == "large-v3"
