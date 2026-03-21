"""Emit JSON describing cached Hugging Face repo roots worth bundling."""

from __future__ import annotations

import json
from pathlib import Path

from config import get_config
from launcher_preflight import (
    REALTIME_MODEL_NAME,
    resolve_hf_file_from_cache,
    resolve_whisper_model_from_cache,
)


def repo_root(value: str | Path) -> str | None:
    """Return the Hugging Face repo cache root for a cached file or snapshot path."""
    path = Path(value).resolve()
    for candidate in (path, *path.parents):
        if candidate.name.startswith("models--"):
            return str(candidate)
    return None


def main() -> int:
    """Print a JSON array of repo cache roots available on this build machine."""
    config = get_config()
    roots: list[str] = []
    whisper_models = [config.transcription.whisper_model]
    if REALTIME_MODEL_NAME != config.transcription.whisper_model:
        whisper_models.append(REALTIME_MODEL_NAME)

    for model_name in whisper_models:
        try:
            root = repo_root(resolve_whisper_model_from_cache(model_name))
        except Exception:
            root = None
        if root and root not in roots:
            roots.append(root)

    if config.grammar.enabled:
        for filename in ("config.json", "pytorch_model.bin"):
            try:
                root = repo_root(resolve_hf_file_from_cache(config.grammar.gector_model, filename))
            except Exception:
                root = None
            if root and root not in roots:
                roots.append(root)

    print(json.dumps(roots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
