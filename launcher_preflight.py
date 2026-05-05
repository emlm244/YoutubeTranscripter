"""Runtime preflight checks used by the Windows launcher."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from config import AppConfig, get_config, get_whisper_models_for_runtime
from grammar_postprocessor import get_languagetool_runtime_status, get_verb_dictionary_path

# Derived from faster_whisper.utils._MODELS to avoid relying on that private API directly.
# Re-sync FASTER_WHISPER_MODEL_REPOS when future faster-whisper releases add or rename models.
FASTER_WHISPER_MODEL_REPOS = {
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "large": "Systran/faster-whisper-large-v3",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
}


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
    from huggingface_hub import try_to_load_from_cache

    repo_id = model_name if "/" in model_name else FASTER_WHISPER_MODEL_REPOS.get(model_name)
    if repo_id is None:
        raise ValueError(f"Unknown Whisper model: {model_name}")

    def _resolve_file(filename: str) -> Path:
        cached_file = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if not isinstance(cached_file, str):
            raise FileNotFoundError(f"{model_name}:{filename}")
        path = Path(cached_file)
        if not path.exists():
            raise FileNotFoundError(f"{model_name}:{filename}")
        return path

    cached_paths = [
        _resolve_file("config.json"),
        _resolve_file("model.bin"),
        _resolve_file("tokenizer.json"),
        _resolve_file("preprocessor_config.json"),
    ]
    last_error: FileNotFoundError | None = None
    for vocabulary_filename in ("vocabulary.json", "vocab.json", "vocab.txt"):
        try:
            cached_paths.append(_resolve_file(vocabulary_filename))
            break
        except FileNotFoundError as exc:
            last_error = exc
    else:
        raise last_error or FileNotFoundError(f"{model_name}:vocabulary")

    return _require_single_snapshot(cached_paths, repo_id=repo_id)


def resolve_hf_file_from_cache(repo_id: str, filename: str) -> Path:
    """Resolve a Hugging Face file from the local cache only."""
    _bootstrap_runtime_environment()
    from huggingface_hub import try_to_load_from_cache

    cached_file = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    if not isinstance(cached_file, str):
        raise FileNotFoundError(f"{repo_id}:{filename}")

    return Path(cached_file)


def _resolve_any_hf_file_from_cache(repo_id: str, filenames: tuple[str, ...]) -> Path:
    last_error: FileNotFoundError | None = None
    for filename in filenames:
        try:
            return resolve_hf_file_from_cache(repo_id, filename)
        except FileNotFoundError as exc:
            last_error = exc
    raise last_error or FileNotFoundError(f"{repo_id}:{'|'.join(filenames)}")


def _require_single_snapshot(paths: list[Path], *, repo_id: str) -> Path:
    snapshot_dirs = {path.parent for path in paths}
    if len(snapshot_dirs) != 1:
        raise FileNotFoundError(f"{repo_id}: cached files span multiple snapshots")
    return paths[0].parent


def _resolve_transformer_model_from_cache(repo_id: str) -> Path:
    """Resolve a transformer repo cache including config, weights, and tokenizer."""
    config_path = resolve_hf_file_from_cache(repo_id, "config.json")
    cached_paths = [
        config_path,
        resolve_hf_file_from_cache(repo_id, "tokenizer_config.json"),
    ]

    try:
        cached_paths.append(resolve_hf_file_from_cache(repo_id, "tokenizer.json"))
    except FileNotFoundError:
        cached_paths.append(resolve_hf_file_from_cache(repo_id, "vocab.json"))
        cached_paths.append(resolve_hf_file_from_cache(repo_id, "merges.txt"))

    cached_paths.append(_resolve_any_hf_file_from_cache(repo_id, ("model.safetensors", "pytorch_model.bin")))

    return _require_single_snapshot(cached_paths, repo_id=repo_id)


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

    try:
        cached_paths = [
            resolve_hf_file_from_cache(model_id, "config.json"),
            _resolve_any_hf_file_from_cache(model_id, ("model.safetensors", "pytorch_model.bin")),
            resolve_hf_file_from_cache(model_id, "tokenizer_config.json"),
        ]
        try:
            cached_paths.append(resolve_hf_file_from_cache(model_id, "tokenizer.json"))
        except FileNotFoundError:
            cached_paths.append(resolve_hf_file_from_cache(model_id, "vocab.json"))
            cached_paths.append(resolve_hf_file_from_cache(model_id, "merges.txt"))
        cached_dir = _require_single_snapshot(cached_paths, repo_id=model_id)
    except Exception:
        return PreflightItem(
            label,
            "info",
            "Not cached yet. First grammar-enhanced run will download these files from Hugging Face.",
        )

    if not all(path.exists() for path in cached_paths):
        return PreflightItem(
            label,
            "info",
            "Some cached files are missing. First grammar-enhanced run may need to re-download them.",
        )

    try:
        config_dict = json.loads(cached_paths[0].read_text(encoding="utf-8"))
    except Exception:
        return PreflightItem(
            label,
            "info",
            "Model config is cached but could not be read. First grammar-enhanced run may need to refresh it.",
        )

    backbone_model_id = config_dict.get("model_id")
    if isinstance(backbone_model_id, str) and backbone_model_id.strip():
        try:
            backbone_dir = _resolve_transformer_model_from_cache(backbone_model_id.strip())
        except Exception:
            return PreflightItem(
                label,
                "info",
                f"GECToR repo is cached, but backbone '{backbone_model_id}' is not cached yet.",
            )
        return PreflightItem(
            label,
            "ok",
            f"Cached in {cached_dir} with backbone cached at {backbone_dir}",
        )

    return PreflightItem(label, "ok", f"Cached in {cached_dir}")


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
) -> list[PreflightItem]:
    """Collect cache/setup notes that matter before the GUI starts."""
    app_config = config or get_config()
    items: list[PreflightItem] = []
    if app_config.transcription.batch_backend in {"local_whisper", "compare"}:
        items.extend(inspect_whisper_model(model_name) for model_name in get_whisper_models_for_runtime(app_config))

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
