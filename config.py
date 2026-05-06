"""Configuration management for the desktop transcription application."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from app_paths import get_config_path

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "large-v3"

WHISPER_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (("large-v3", "Whisper Large v3"),)

BATCH_BACKEND_OPTIONS: tuple[tuple[str, str], ...] = (
    ("openai", "OpenAI"),
    ("local_whisper", "Local Whisper"),
    ("compare", "Compare"),
)

OPENAI_BATCH_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("gpt-4o-transcribe", "GPT-4o Transcribe"),
    ("gpt-4o-mini-transcribe", "GPT-4o mini Transcribe"),
)

DEVICE_PREFERENCE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("cuda", "Prefer GPU"),
    ("cpu", "Force CPU"),
)

COMPUTE_TYPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("float16", "Float16"),
    ("int8", "Int8"),
)

GRAMMAR_BACKEND_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("gector", "GECToR"),
    ("languagetool", "LanguageTool"),
)

# Normalize retired shortcut/distil selections onto the single supported local Whisper model.
# We keep the aliases for saved configs, then emit a warning during load because speed characteristics can change.
_WHISPER_MODEL_ALIASES: dict[str, str] = {
    "distil-large-v2": DEFAULT_WHISPER_MODEL,
    "distil-large-v3": DEFAULT_WHISPER_MODEL,
    "distil-large-v3.5": DEFAULT_WHISPER_MODEL,
    "distil-medium.en": DEFAULT_WHISPER_MODEL,
    "distil-small.en": DEFAULT_WHISPER_MODEL,
    "large": DEFAULT_WHISPER_MODEL,
}


def normalize_whisper_model_name(value: object, default: str = DEFAULT_WHISPER_MODEL) -> str:
    """Normalize legacy or simplified Whisper model selections."""
    if not isinstance(value, str):
        return default

    model_name = value.strip()
    if not model_name:
        return default

    normalized_model_name = _WHISPER_MODEL_ALIASES.get(model_name, model_name)
    known_models = {option for option, _label in WHISPER_MODEL_OPTIONS}
    return normalized_model_name if normalized_model_name in known_models else default


def _normalize_option(value: object, *, allowed: set[str], default: str) -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    return normalized if normalized in allowed else default


def normalize_batch_backend(value: object, default: str = "openai") -> str:
    return _normalize_option(
        value,
        allowed={option for option, _label in BATCH_BACKEND_OPTIONS},
        default=default,
    )


def normalize_openai_batch_model(value: object, default: str = "gpt-4o-transcribe") -> str:
    if not isinstance(value, str):
        return default
    model = value.strip()
    if not model:
        return default
    known_models = {option for option, _label in OPENAI_BATCH_MODEL_OPTIONS}
    return model if model in known_models else default


@dataclass
class TranscriptionConfig:
    """Configuration for transcription settings."""

    batch_backend: str = "openai"  # "openai", "local_whisper", "compare"
    openai_batch_model: str = "gpt-4o-transcribe"
    whisper_model: str = DEFAULT_WHISPER_MODEL  # Best accuracy for accented/meeting speech
    device_preference: str = "auto"  # "auto", "cuda", "cpu"
    compute_type: str = "auto"  # "auto", "float16", "int8"
    beam_size: int = 5
    temperature: float = 0.0
    vad_filter: bool = True
    # Whisper's no-speech confidence threshold: lower is more permissive for real-world recordings.
    no_speech_threshold: float = 0.3
    batch_size: int = 32  # Optimized for high-end GPU with 24GB+ VRAM
    cpu_fallback_batch_size: int = 8
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3

    # Advanced accuracy settings
    word_timestamps: bool = True  # Enable word-level timestamps for precise alignment
    patience: float = 1.0  # Beam search patience factor (higher = more thorough)
    length_penalty: float = 1.0  # Penalty for shorter/longer sequences
    hallucination_silence_threshold: float = 0.5  # Silence threshold to prevent hallucinations

    # VAD tuning for accuracy (tuned for accented/meeting speech)
    vad_threshold: float = 0.25  # Lower = more sensitive to quieter/accented speech
    min_speech_duration_ms: int = 50  # Catch brief accented syllables
    min_silence_duration_ms: int = 2000  # Don't cut during thinking pauses
    speech_pad_ms: int = 400  # Catch trailing syllables

    # Disabled by default because long prompts can leak into the transcript itself.
    initial_prompt: Optional[str] = None
    language: Optional[str] = None  # Force language (None = auto-detect)
    hotwords: Optional[str] = None  # Domain vocabulary hints (names, jargon)
    condition_on_previous_text: bool = False  # Reduce error carry-over across pauses/segments
    clean_filler_words: bool = False  # Off by default for raw-faithful output
    filter_hallucinations: bool = True  # Drop common silence/music hallucinations by default
    deduplicate_repeated_segments: bool = True  # Drop obvious decode loops by default

    # Audio preprocessing
    noise_reduction_enabled: bool = True  # Spectral noise reduction before Whisper
    normalize_audio: bool = True  # Loudness normalization before Whisper


@dataclass
class RecordingConfig:
    """Configuration for audio recording."""

    default_microphone: str = ""
    sample_rate: int = 16000


@dataclass
class UIConfig:
    """Configuration for UI settings."""

    last_youtube_url: str = ""
    output_format: str = "plain"
    transcription_preset: str = "max_accuracy"


@dataclass
class GrammarConfig:
    """Configuration for grammar post-processing.

    Supports GECToR (primary, GPU-accelerated) and LanguageTool (fallback).
    """

    enabled: bool = False  # Optional post-processing; off by default for faithful output
    language: str = "en-US"  # Language for grammar checking
    backend: str = "auto"  # "auto", "gector", "languagetool"
    gector_model: str = "gotutiyan/gector-roberta-large-5k"
    gector_batch_size: int = 8
    gector_iterations: int = 5


@dataclass
class AccuracyPreset:
    """Predefined accuracy profile for transcription."""

    name: str
    whisper_model: str
    beam_size: int
    batch_size: int
    initial_prompt: Optional[str]
    language: Optional[str]
    temperature: float
    vad_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    condition_on_previous_text: bool
    filter_hallucinations: bool
    deduplicate_repeated_segments: bool


_LEGACY_INITIAL_PROMPTS = frozenset(
    {
        (
            "This is a faithful transcript of spoken audio. Preserve the speaker's exact wording, "
            "filler words, repetitions, and unfinished thoughts. Do not paraphrase, summarize, or "
            "rewrite for grammar. Use punctuation and capitalization only to reflect the spoken words."
        ),
        (
            "This is a faithful transcript of spoken audio, including meetings with diverse accents. "
            "Preserve exact wording, filler words, repetitions, and uncertain phrasing. "
            "Do not paraphrase or rewrite for grammar. Use punctuation and capitalization only to "
            "make the spoken words readable."
        ),
    }
)

ACCURACY_PRESETS: dict[str, AccuracyPreset] = {
    "speed": AccuracyPreset(
        name="Speed",
        whisper_model=DEFAULT_WHISPER_MODEL,
        beam_size=2,
        batch_size=48,
        initial_prompt=None,
        language=None,
        temperature=0.0,
        vad_threshold=0.35,
        min_speech_duration_ms=100,
        min_silence_duration_ms=1500,
        speech_pad_ms=300,
        condition_on_previous_text=False,
        filter_hallucinations=True,
        deduplicate_repeated_segments=True,
    ),
    "balanced": AccuracyPreset(
        name="Balanced",
        whisper_model=DEFAULT_WHISPER_MODEL,
        beam_size=4,
        batch_size=40,
        initial_prompt=None,
        language="en",
        temperature=0.0,
        vad_threshold=0.30,
        min_speech_duration_ms=50,
        min_silence_duration_ms=2000,
        speech_pad_ms=400,
        condition_on_previous_text=False,
        filter_hallucinations=True,
        deduplicate_repeated_segments=True,
    ),
    "max_accuracy": AccuracyPreset(
        name="Maximum Accuracy",
        whisper_model=DEFAULT_WHISPER_MODEL,
        beam_size=5,
        batch_size=32,
        initial_prompt=None,
        language="en",
        temperature=0.0,
        vad_threshold=0.25,
        min_speech_duration_ms=50,
        min_silence_duration_ms=2000,
        speech_pad_ms=400,
        condition_on_previous_text=False,
        filter_hallucinations=True,
        deduplicate_repeated_segments=True,
    ),
}

DEFAULT_PRESET = "max_accuracy"


def apply_preset(config: TranscriptionConfig, preset_name: str) -> None:
    """Apply an accuracy preset to a TranscriptionConfig in-place."""
    preset = ACCURACY_PRESETS.get(preset_name)
    if preset is None:
        return
    config.whisper_model = preset.whisper_model
    config.beam_size = preset.beam_size
    config.batch_size = preset.batch_size
    config.initial_prompt = preset.initial_prompt
    config.language = preset.language
    config.temperature = preset.temperature
    config.vad_threshold = preset.vad_threshold
    config.min_speech_duration_ms = preset.min_speech_duration_ms
    config.min_silence_duration_ms = preset.min_silence_duration_ms
    config.speech_pad_ms = preset.speech_pad_ms
    config.condition_on_previous_text = preset.condition_on_previous_text
    config.filter_hallucinations = preset.filter_hallucinations
    config.deduplicate_repeated_segments = preset.deduplicate_repeated_segments


def _normalize_prompt_whitespace(value: str) -> str:
    """Collapse internal whitespace so legacy prompt matching is stable."""
    return " ".join(value.split())


_NORMALIZED_LEGACY_INITIAL_PROMPTS = frozenset(_normalize_prompt_whitespace(prompt) for prompt in _LEGACY_INITIAL_PROMPTS)


def _coerce_initial_prompt(value: object, default: Optional[str]) -> Optional[str]:
    """Normalize saved prompt values and disable legacy prompt injection defaults."""
    if value is None:
        return None
    if not isinstance(value, str):
        return default

    cleaned = _normalize_prompt_whitespace(value)
    if not cleaned:
        return None

    if cleaned in _NORMALIZED_LEGACY_INITIAL_PROMPTS:
        return None

    return value.strip()


def _is_legacy_initial_prompt(value: object) -> bool:
    """Return whether a saved prompt matches one of the retired instruction prompts."""
    if not isinstance(value, str):
        return False
    cleaned = _normalize_prompt_whitespace(value)
    return cleaned in _NORMALIZED_LEGACY_INITIAL_PROMPTS


@dataclass
class AppConfig:
    """Main application configuration."""

    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    grammar: GrammarConfig = field(default_factory=GrammarConfig)

    # System settings - VRAM budget for high-end GPU
    # gpu_memory_fraction is a best-effort cap for PyTorch components (CTranslate2 manages its own memory)
    gpu_memory_fraction: float = 0.90  # Optimized for 24GB+ GPU
    max_audio_size_mb: int = 500
    max_filename_length: int = 200

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if not (0.0 < self.gpu_memory_fraction <= 1.0):
            raise ValueError(f"gpu_memory_fraction must be between 0 and 1, got {self.gpu_memory_fraction}")
        if self.max_audio_size_mb <= 0:
            raise ValueError(f"max_audio_size_mb must be positive, got {self.max_audio_size_mb}")
        if self.max_filename_length <= 0:
            raise ValueError(f"max_filename_length must be positive, got {self.max_filename_length}")

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save config. Defaults to config.json in app directory.
        """
        if path is None:
            path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "transcription": asdict(self.transcription),
            "recording": asdict(self.recording),
            "ui": asdict(self.ui),
            "grammar": asdict(self.grammar),
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "max_audio_size_mb": self.max_audio_size_mb,
            "max_filename_length": self.max_filename_length,
        }

        temp_fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Optional[Path] = None) -> AppConfig:
        """Load configuration from JSON file.

        Args:
            path: Path to load config from. Defaults to config.json in app directory.

        Returns:
            AppConfig instance with loaded values or defaults.
        """
        if path is None:
            path = get_config_path()

        if not path.exists():
            return cls()

        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return cls()

            config = cls()

            def _coerce_fraction(value: object, default: float) -> float:
                if isinstance(value, bool):
                    return default
                if not isinstance(value, (int, float, str)):
                    return default
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return default
                if 0.0 < numeric <= 1.0:
                    return numeric
                return default

            def _coerce_positive_int(value: object, default: int) -> int:
                if isinstance(value, bool):
                    return default
                if not isinstance(value, (int, float, str)):
                    return default
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    return default
                if numeric > 0:
                    return numeric
                return default

            def _coerce_nonnegative_int(value: object, default: int) -> int:
                if isinstance(value, bool):
                    return default
                if not isinstance(value, (int, float, str)):
                    return default
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    return default
                if numeric >= 0:
                    return numeric
                return default

            def _coerce_float(value: object, default: float) -> float:
                if isinstance(value, bool):
                    return default
                if not isinstance(value, (int, float, str)):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            def _coerce_bool(value: object, default: bool) -> bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)) and value in {0, 1}:
                    return bool(value)
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized in {"1", "true", "yes", "on"}:
                        return True
                    if normalized in {"0", "false", "no", "off"}:
                        return False
                return default

            def _coerce_str(value: object, default: str) -> str:
                if not isinstance(value, str):
                    return default
                stripped = value.strip()
                return stripped if stripped else default

            def _coerce_optional_str(value: object, default: str | None) -> str | None:
                if value is None:
                    return default
                if not isinstance(value, str):
                    return default
                stripped = value.strip()
                return stripped if stripped else None

            # Load transcription settings
            t = data.get("transcription")
            if isinstance(t, dict):
                saved_whisper_model = t.get("whisper_model", config.transcription.whisper_model)
                normalized_whisper_model = normalize_whisper_model_name(
                    saved_whisper_model,
                    default=config.transcription.whisper_model,
                )
                if isinstance(saved_whisper_model, str):
                    original_model = saved_whisper_model.strip()
                    if original_model and normalized_whisper_model != original_model:
                        logger.warning(
                            "Legacy Whisper model '%s' has been migrated to '%s'. "
                            "This keeps old configs loading, but may change transcription speed.",
                            original_model,
                            normalized_whisper_model,
                        )
                saved_initial_prompt = t.get("initial_prompt", config.transcription.initial_prompt)
                legacy_prompt_migration = _is_legacy_initial_prompt(saved_initial_prompt)
                config.transcription = TranscriptionConfig(
                    batch_backend=normalize_batch_backend(
                        # Keep existing configs on local Whisper unless they opt into OpenAI.
                        t.get("batch_backend", "local_whisper"),
                        default="local_whisper",
                    ),
                    openai_batch_model=normalize_openai_batch_model(
                        t.get("openai_batch_model", config.transcription.openai_batch_model),
                        default=config.transcription.openai_batch_model,
                    ),
                    whisper_model=normalized_whisper_model,
                    device_preference=_normalize_option(
                        t.get("device_preference", config.transcription.device_preference),
                        allowed={value for value, _label in DEVICE_PREFERENCE_OPTIONS},
                        default=config.transcription.device_preference,
                    ),
                    compute_type=_normalize_option(
                        t.get("compute_type", config.transcription.compute_type),
                        allowed={value for value, _label in COMPUTE_TYPE_OPTIONS},
                        default=config.transcription.compute_type,
                    ),
                    beam_size=_coerce_positive_int(t.get("beam_size", config.transcription.beam_size), config.transcription.beam_size),
                    temperature=_coerce_float(t.get("temperature", config.transcription.temperature), config.transcription.temperature),
                    vad_filter=_coerce_bool(t.get("vad_filter", config.transcription.vad_filter), config.transcription.vad_filter),
                    no_speech_threshold=_coerce_float(
                        t.get("no_speech_threshold", config.transcription.no_speech_threshold),
                        config.transcription.no_speech_threshold,
                    ),
                    batch_size=_coerce_positive_int(t.get("batch_size", config.transcription.batch_size), config.transcription.batch_size),
                    cpu_fallback_batch_size=_coerce_positive_int(
                        t.get("cpu_fallback_batch_size", config.transcription.cpu_fallback_batch_size),
                        config.transcription.cpu_fallback_batch_size,
                    ),
                    repetition_penalty=_coerce_float(
                        t.get("repetition_penalty", config.transcription.repetition_penalty),
                        config.transcription.repetition_penalty,
                    ),
                    no_repeat_ngram_size=_coerce_nonnegative_int(
                        t.get("no_repeat_ngram_size", config.transcription.no_repeat_ngram_size),
                        config.transcription.no_repeat_ngram_size,
                    ),
                    # Advanced accuracy settings
                    word_timestamps=_coerce_bool(
                        t.get("word_timestamps", config.transcription.word_timestamps),
                        config.transcription.word_timestamps,
                    ),
                    patience=_coerce_float(t.get("patience", config.transcription.patience), config.transcription.patience),
                    length_penalty=_coerce_float(
                        t.get("length_penalty", config.transcription.length_penalty),
                        config.transcription.length_penalty,
                    ),
                    hallucination_silence_threshold=_coerce_float(
                        t.get(
                            "hallucination_silence_threshold",
                            config.transcription.hallucination_silence_threshold,
                        ),
                        config.transcription.hallucination_silence_threshold,
                    ),
                    # VAD tuning
                    vad_threshold=_coerce_float(
                        t.get("vad_threshold", config.transcription.vad_threshold), config.transcription.vad_threshold
                    ),
                    min_speech_duration_ms=_coerce_nonnegative_int(
                        t.get("min_speech_duration_ms", config.transcription.min_speech_duration_ms),
                        config.transcription.min_speech_duration_ms,
                    ),
                    min_silence_duration_ms=_coerce_nonnegative_int(
                        t.get("min_silence_duration_ms", config.transcription.min_silence_duration_ms),
                        config.transcription.min_silence_duration_ms,
                    ),
                    speech_pad_ms=_coerce_nonnegative_int(
                        t.get("speech_pad_ms", config.transcription.speech_pad_ms),
                        config.transcription.speech_pad_ms,
                    ),
                    # Accuracy fields
                    initial_prompt=_coerce_initial_prompt(
                        saved_initial_prompt,
                        config.transcription.initial_prompt,
                    ),
                    language=_coerce_optional_str(t.get("language", config.transcription.language), config.transcription.language),
                    hotwords=_coerce_optional_str(t.get("hotwords", config.transcription.hotwords), config.transcription.hotwords),
                    condition_on_previous_text=(
                        False
                        if legacy_prompt_migration
                        else _coerce_bool(
                            t.get(
                                "condition_on_previous_text",
                                config.transcription.condition_on_previous_text,
                            ),
                            config.transcription.condition_on_previous_text,
                        )
                    ),
                    clean_filler_words=_coerce_bool(
                        t.get("clean_filler_words", config.transcription.clean_filler_words),
                        config.transcription.clean_filler_words,
                    ),
                    filter_hallucinations=(
                        True
                        if legacy_prompt_migration
                        else _coerce_bool(
                            t.get(
                                "filter_hallucinations",
                                config.transcription.filter_hallucinations,
                            ),
                            config.transcription.filter_hallucinations,
                        )
                    ),
                    deduplicate_repeated_segments=(
                        True
                        if legacy_prompt_migration
                        else _coerce_bool(
                            t.get(
                                "deduplicate_repeated_segments",
                                config.transcription.deduplicate_repeated_segments,
                            ),
                            config.transcription.deduplicate_repeated_segments,
                        )
                    ),
                    # Audio preprocessing
                    noise_reduction_enabled=_coerce_bool(
                        t.get(
                            "noise_reduction_enabled",
                            config.transcription.noise_reduction_enabled,
                        ),
                        config.transcription.noise_reduction_enabled,
                    ),
                    normalize_audio=_coerce_bool(
                        t.get("normalize_audio", config.transcription.normalize_audio),
                        config.transcription.normalize_audio,
                    ),
                )

            # Load recording settings
            r = data.get("recording")
            if isinstance(r, dict):
                config.recording = RecordingConfig(
                    default_microphone=_coerce_str(
                        r.get("default_microphone", config.recording.default_microphone),
                        config.recording.default_microphone,
                    ),
                    sample_rate=_coerce_positive_int(
                        r.get("sample_rate", config.recording.sample_rate),
                        config.recording.sample_rate,
                    ),
                )

            # Load UI settings
            u = data.get("ui")
            if isinstance(u, dict):
                config.ui = UIConfig(
                    last_youtube_url=_coerce_str(
                        u.get("last_youtube_url", config.ui.last_youtube_url),
                        config.ui.last_youtube_url,
                    ),
                    output_format=_coerce_str(
                        u.get("output_format", config.ui.output_format),
                        config.ui.output_format,
                    ),
                    transcription_preset=_coerce_str(
                        u.get("transcription_preset", config.ui.transcription_preset),
                        config.ui.transcription_preset,
                    ),
                )

            # Load grammar settings
            g = data.get("grammar")
            if isinstance(g, dict):
                grammar_backend = _normalize_option(
                    g.get("backend", config.grammar.backend),
                    allowed={value for value, _label in GRAMMAR_BACKEND_OPTIONS},
                    default=config.grammar.backend,
                )

                config.grammar = GrammarConfig(
                    enabled=_coerce_bool(g.get("enabled", config.grammar.enabled), config.grammar.enabled),
                    language=_coerce_str(g.get("language", config.grammar.language), config.grammar.language),
                    backend=grammar_backend,
                    gector_model=_coerce_str(g.get("gector_model", config.grammar.gector_model), config.grammar.gector_model),
                    gector_batch_size=_coerce_positive_int(
                        g.get("gector_batch_size", config.grammar.gector_batch_size),
                        config.grammar.gector_batch_size,
                    ),
                    gector_iterations=_coerce_positive_int(
                        g.get("gector_iterations", config.grammar.gector_iterations),
                        config.grammar.gector_iterations,
                    ),
                )

            if config.transcription.device_preference not in {value for value, _ in DEVICE_PREFERENCE_OPTIONS}:
                config.transcription.device_preference = "auto"
            if config.transcription.compute_type not in {value for value, _ in COMPUTE_TYPE_OPTIONS}:
                config.transcription.compute_type = "auto"

            # Load system settings - VRAM budget
            config.gpu_memory_fraction = _coerce_fraction(
                data.get("gpu_memory_fraction", config.gpu_memory_fraction),
                config.gpu_memory_fraction,
            )
            config.max_audio_size_mb = _coerce_positive_int(
                data.get("max_audio_size_mb", config.max_audio_size_mb),
                config.max_audio_size_mb,
            )
            config.max_filename_length = _coerce_positive_int(
                data.get("max_filename_length", config.max_filename_length),
                config.max_filename_length,
            )

            return config

        except json.JSONDecodeError as exc:
            logger.warning("Config file contains invalid JSON, using defaults: %s", exc)
            return cls()
        except (TypeError, ValueError) as exc:
            logger.warning("Config file contains invalid value types, using defaults: %s", exc)
            return cls()


# Global config instance with thread-safe initialization
_config: Optional[AppConfig] = None
_config_lock = threading.Lock()


def get_config() -> AppConfig:
    """Get the global configuration instance (thread-safe)."""
    global _config
    if _config is None:
        with _config_lock:
            # Double-check pattern to avoid redundant lock acquisition
            if _config is None:
                _config = AppConfig.load()
    return _config


def save_config() -> None:
    """Save the global configuration."""
    if _config is not None:
        _config.save()


def get_whisper_models_for_runtime(config: AppConfig | None = None) -> list[str]:
    """Return the active and preset Whisper models that users can realistically select."""
    app_config = config or get_config()
    ordered_models: list[str] = []

    def _append(model_name: str | None) -> None:
        if model_name and model_name not in ordered_models:
            ordered_models.append(model_name)

    _append(normalize_whisper_model_name(app_config.transcription.whisper_model))
    for preset in ACCURACY_PRESETS.values():
        _append(normalize_whisper_model_name(preset.whisper_model))

    return ordered_models
