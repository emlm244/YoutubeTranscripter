"""Configuration management for YouTube Transcriber application."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from app_paths import get_config_path


@dataclass
class TranscriptionConfig:
    """Configuration for transcription settings."""

    whisper_model: str = "large-v3"  # Best accuracy for accented/meeting speech
    beam_size: int = 5
    temperature: float = 0.0
    vad_filter: bool = True
    # Whisper's no-speech confidence threshold: lower is more permissive for real-world recordings.
    no_speech_threshold: float = 0.3
    batch_size: int = 32  # Optimized for high-end GPU with 24GB+ VRAM
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

    # Accuracy fields for accented/meeting speech
    initial_prompt: Optional[str] = None  # Whisper context prompt for guiding transcription
    language: Optional[str] = None  # Force language (None = auto-detect)
    hotwords: Optional[str] = None  # Domain vocabulary hints (names, jargon)
    condition_on_previous_text: bool = True  # Context flow between segments
    clean_filler_words: bool = True  # Remove umm/uh/you know (ON by default)

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

    theme: str = "dark"
    accent_color: str = "blue"
    window_width: int = 1280
    window_height: int = 1024
    splitter_ratios: tuple[float, float, float] = (0.4, 0.2, 0.4)
    remember_window_position: bool = True
    last_youtube_url: str = ""
    output_format: str = "plain"
    transcription_preset: str = "max_accuracy"


@dataclass
class GrammarConfig:
    """Configuration for grammar post-processing.

    Supports GECToR (primary, GPU-accelerated) and LanguageTool (fallback).
    """

    enabled: bool = True  # ON by default
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
    initial_prompt: Optional[str]
    language: Optional[str]
    temperature: float
    vad_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int


# Accent-aware prompt for balanced/accuracy presets
_ACCENT_PROMPT = (
    "This is a professional meeting transcript. Some speakers have non-native English accents "
    "including Indian, Eastern European, and other international accents. "
    "Use proper grammar, punctuation, and capitalization."
)

ACCURACY_PRESETS: dict[str, AccuracyPreset] = {
    "speed": AccuracyPreset(
        name="Speed",
        whisper_model="distil-large-v3",
        initial_prompt=None,
        language=None,
        temperature=0.0,
        vad_threshold=0.35,
        min_speech_duration_ms=100,
        min_silence_duration_ms=1500,
        speech_pad_ms=300,
    ),
    "balanced": AccuracyPreset(
        name="Balanced",
        whisper_model="distil-large-v3",
        initial_prompt=_ACCENT_PROMPT,
        language="en",
        temperature=0.0,
        vad_threshold=0.30,
        min_speech_duration_ms=50,
        min_silence_duration_ms=2000,
        speech_pad_ms=400,
    ),
    "max_accuracy": AccuracyPreset(
        name="Maximum Accuracy",
        whisper_model="large-v3",
        initial_prompt=_ACCENT_PROMPT,
        language="en",
        temperature=0.0,
        vad_threshold=0.25,
        min_speech_duration_ms=50,
        min_silence_duration_ms=2000,
        speech_pad_ms=400,
    ),
}

DEFAULT_PRESET = "max_accuracy"


def apply_preset(config: TranscriptionConfig, preset_name: str) -> None:
    """Apply an accuracy preset to a TranscriptionConfig in-place."""
    preset = ACCURACY_PRESETS.get(preset_name)
    if preset is None:
        return
    config.whisper_model = preset.whisper_model
    config.initial_prompt = preset.initial_prompt
    config.language = preset.language
    config.temperature = preset.temperature
    config.vad_threshold = preset.vad_threshold
    config.min_speech_duration_ms = preset.min_speech_duration_ms
    config.min_silence_duration_ms = preset.min_silence_duration_ms
    config.speech_pad_ms = preset.speech_pad_ms


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
            raise ValueError(
                f"gpu_memory_fraction must be between 0 and 1, got {self.gpu_memory_fraction}"
            )
        if self.max_audio_size_mb <= 0:
            raise ValueError(
                f"max_audio_size_mb must be positive, got {self.max_audio_size_mb}"
            )
        if self.max_filename_length <= 0:
            raise ValueError(
                f"max_filename_length must be positive, got {self.max_filename_length}"
            )

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save config. Defaults to config.json in app directory.
        """
        if path is None:
            path = get_config_path()

        data = {
            "transcription": asdict(self.transcription),
            "recording": asdict(self.recording),
            "ui": {**asdict(self.ui), "splitter_ratios": list(self.ui.splitter_ratios)},
            "grammar": asdict(self.grammar),
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "max_audio_size_mb": self.max_audio_size_mb,
            "max_filename_length": self.max_filename_length,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
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
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

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

            # Load transcription settings
            if "transcription" in data:
                t = data["transcription"]
                config.transcription = TranscriptionConfig(
                    whisper_model=t.get("whisper_model", "large-v3"),
                    beam_size=t.get("beam_size", 5),
                    temperature=t.get("temperature", 0.0),
                    vad_filter=t.get("vad_filter", True),
                    no_speech_threshold=t.get("no_speech_threshold", 0.3),
                    batch_size=t.get("batch_size", 32),
                    repetition_penalty=t.get("repetition_penalty", 1.2),
                    no_repeat_ngram_size=t.get("no_repeat_ngram_size", 3),
                    # Advanced accuracy settings
                    word_timestamps=t.get("word_timestamps", True),
                    patience=t.get("patience", 1.0),
                    length_penalty=t.get("length_penalty", 1.0),
                    hallucination_silence_threshold=t.get("hallucination_silence_threshold", 0.5),
                    # VAD tuning
                    vad_threshold=t.get("vad_threshold", 0.25),
                    min_speech_duration_ms=t.get("min_speech_duration_ms", 50),
                    min_silence_duration_ms=t.get("min_silence_duration_ms", 2000),
                    speech_pad_ms=t.get("speech_pad_ms", 400),
                    # Accuracy fields
                    initial_prompt=t.get("initial_prompt", None),
                    language=t.get("language", None),
                    hotwords=t.get("hotwords", None),
                    condition_on_previous_text=t.get("condition_on_previous_text", True),
                    clean_filler_words=t.get("clean_filler_words", True),
                    # Audio preprocessing
                    noise_reduction_enabled=t.get("noise_reduction_enabled", True),
                    normalize_audio=t.get("normalize_audio", True),
                )

            # Load recording settings
            if "recording" in data:
                r = data["recording"]
                config.recording = RecordingConfig(
                    default_microphone=r.get("default_microphone", ""),
                    sample_rate=r.get("sample_rate", 16000),
                )

            # Load UI settings
            if "ui" in data:
                u = data["ui"]
                ratios = u.get("splitter_ratios", [0.4, 0.2, 0.4])
                config.ui = UIConfig(
                    theme=u.get("theme", "dark"),
                    accent_color=u.get("accent_color", "blue"),
                    window_width=u.get("window_width", 1280),
                    window_height=u.get("window_height", 1024),
                    splitter_ratios=tuple(ratios) if isinstance(ratios, list) else ratios,
                    remember_window_position=u.get("remember_window_position", True),
                    last_youtube_url=u.get("last_youtube_url", ""),
                    output_format=u.get("output_format", "plain"),
                    transcription_preset=u.get("transcription_preset", "max_accuracy"),
                )

            # Load grammar settings
            if "grammar" in data:
                g = data["grammar"]
                config.grammar = GrammarConfig(
                    enabled=g.get("enabled", True),
                    language=g.get("language", "en-US"),
                    backend=g.get("backend", "auto"),
                    gector_model=g.get("gector_model", "gotutiyan/gector-roberta-large-5k"),
                    gector_batch_size=g.get("gector_batch_size", 8),
                    gector_iterations=g.get("gector_iterations", 5),
                )

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

        except (json.JSONDecodeError, KeyError, TypeError):
            # Return defaults on error
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
    global _config
    if _config is not None:
        _config.save()

