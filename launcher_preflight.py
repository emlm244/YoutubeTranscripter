"""Runtime preflight checks used by the Windows launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config import AppConfig, get_config
from grammar_postprocessor import get_languagetool_runtime_status, get_verb_dictionary_path

REALTIME_MODEL_NAME = "distil-large-v3"


@dataclass(frozen=True)
class PreflightItem:
    """A single user-facing runtime preflight result."""

    label: str
    status: Literal["ok", "info", "warning"]
    detail: str


def _bootstrap_runtime_environment() -> None:
    """Apply the shared runtime bootstrap only when cache probes actually run."""
    from runtime_bootstrap import bootstrap_runtime

    bootstrap_runtime()


def resolve_whisper_model_from_cache(model_name: str) -> Path:
    """Resolve a faster-whisper model from the local cache only."""
    _bootstrap_runtime_environment()
    from faster_whisper.utils import _MODELS as faster_whisper_models
    from huggingface_hub import try_to_load_from_cache

    repo_id = model_name if "/" in model_name else faster_whisper_models.get(model_name)
    if repo_id is None:
        raise ValueError(f"Unknown Whisper model: {model_name}")

    cached_config = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    if not isinstance(cached_config, str):
        raise FileNotFoundError(model_name)

    return Path(cached_config).parent


def resolve_hf_file_from_cache(repo_id: str, filename: str) -> Path:
    """Resolve a Hugging Face file from the local cache only."""
    _bootstrap_runtime_environment()
    from huggingface_hub import try_to_load_from_cache

    cached_file = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    if not isinstance(cached_file, str):
        raise FileNotFoundError(f"{repo_id}:{filename}")

    return Path(cached_file)


def inspect_whisper_model(model_name: str) -> PreflightItem:
    """Report whether a faster-whisper model is already cached locally."""
    label = f"Whisper model '{model_name}'"
    try:
        model_path = resolve_whisper_model_from_cache(model_name)
    except Exception:
        return PreflightItem(
            label,
            "info",
            "Not cached yet. First use will download this model from Hugging Face.",
        )

    if model_path.exists():
        return PreflightItem(label, "ok", f"Cached at {model_path}")

    return PreflightItem(
        label,
        "info",
        "Cache lookup returned no local model directory. First use will download it.",
    )


def inspect_gector_model(model_id: str) -> PreflightItem:
    """Report whether the configured GECToR model files are already cached."""
    label = f"Grammar model '{model_id}'"
    required_files = ("config.json", "pytorch_model.bin")

    try:
        cached_paths = [resolve_hf_file_from_cache(model_id, filename) for filename in required_files]
    except Exception:
        return PreflightItem(
            label,
            "info",
            "Not cached yet. First grammar-enhanced run will download these files from Hugging Face.",
        )

    if all(path.exists() for path in cached_paths):
        return PreflightItem(label, "ok", f"Cached in {cached_paths[0].parent}")

    return PreflightItem(
        label,
        "info",
        "Some cached files are missing. First grammar-enhanced run may need to re-download them.",
    )


def inspect_verb_dictionary() -> PreflightItem:
    """Report whether the bundled GECToR verb dictionary is present."""
    verb_dict_path = get_verb_dictionary_path()
    if verb_dict_path is not None and verb_dict_path.exists():
        return PreflightItem("GECToR verb dictionary", "ok", f"Available at {verb_dict_path}")

    return PreflightItem(
        "GECToR verb dictionary",
        "info",
        "Not bundled yet. First grammar-enhanced run will download it.",
    )


def inspect_language_tool_runtime() -> PreflightItem:
    """Report the expected LanguageTool first-run behavior."""
    runtime_status = get_languagetool_runtime_status()
    if runtime_status.available:
        return PreflightItem("LanguageTool fallback", "ok", runtime_status.detail)
    if runtime_status.state == "missing_assets":
        return PreflightItem(
            "LanguageTool fallback",
            "info",
            "Not cached yet. First fallback use will download local LanguageTool assets.",
        )

    return PreflightItem(
        "LanguageTool fallback",
        "warning",
        runtime_status.detail,
    )


def collect_preflight_items(
    config: AppConfig | None = None,
    *,
    realtime_model_name: str = REALTIME_MODEL_NAME,
) -> list[PreflightItem]:
    """Collect cache/setup notes that matter before the GUI starts."""
    app_config = config or get_config()
    items = [inspect_whisper_model(app_config.transcription.whisper_model)]

    if realtime_model_name != app_config.transcription.whisper_model:
        items.append(inspect_whisper_model(realtime_model_name))

    if app_config.grammar.enabled:
        items.append(inspect_gector_model(app_config.grammar.gector_model))
        items.append(inspect_verb_dictionary())
        items.append(inspect_language_tool_runtime())

    return items


def format_item(item: PreflightItem) -> str:
    """Render a preflight item with launcher-friendly prefixes."""
    prefix = {
        "ok": "[OK]",
        "info": "[INFO]",
        "warning": "[WARNING]",
    }[item.status]
    return f"{prefix} {item.label}: {item.detail}"


def main() -> int:
    """Print a concise runtime preflight summary for the batch launcher."""
    print("[INFO] Runtime asset preflight:")
    for item in collect_preflight_items():
        print(format_item(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

