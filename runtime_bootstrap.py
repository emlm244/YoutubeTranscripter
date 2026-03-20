"""Shared import-safe runtime bootstrap for source and packaged runs."""

from __future__ import annotations

import os
import platform

from app_paths import configure_runtime_environment

_BOOTSTRAPPED = False


def bootstrap_runtime() -> None:
    """Apply environment tweaks before heavy runtime dependencies import."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    if platform.system() == "Windows":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

    configure_runtime_environment()
    _BOOTSTRAPPED = True


bootstrap_runtime()
