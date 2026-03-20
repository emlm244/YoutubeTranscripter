"""Best-effort torch import helpers for packaged and source runs."""

from __future__ import annotations

import logging
from typing import Any

from app_paths import preload_windows_torch_dependencies, write_startup_diagnostics

logger = logging.getLogger(__name__)

_TORCH_MODULE: Any | None = None
_TORCH_IMPORT_ERROR: BaseException | None = None


def get_torch(*, context: str) -> Any | None:
    """Return torch when available, else None after recording diagnostics once."""
    global _TORCH_MODULE, _TORCH_IMPORT_ERROR

    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    if _TORCH_IMPORT_ERROR is not None:
        return None

    preload_notes = preload_windows_torch_dependencies()

    try:
        import torch as torch_module
    except Exception as exc:
        _TORCH_IMPORT_ERROR = exc
        diagnostic_path = write_startup_diagnostics(
            context=f"{context}:import_torch",
            error=exc,
            notes=preload_notes,
        )
        logger.warning("PyTorch unavailable; torch-backed features will be limited: %s", exc)
        if diagnostic_path is not None:
            logger.warning("Startup diagnostics written to: %s", diagnostic_path)
        return None

    _TORCH_MODULE = torch_module
    return _TORCH_MODULE


def get_torch_import_error() -> BaseException | None:
    """Return the cached torch import error, if torch failed to import."""
    return _TORCH_IMPORT_ERROR
