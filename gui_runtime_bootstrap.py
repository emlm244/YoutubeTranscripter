"""GUI-specific runtime bootstrap.

Imports torch before Qt on Windows to avoid DLL-resolution conflicts between
PyTorch and the Qt runtime.
"""

from __future__ import annotations

import os
import sys

from runtime_bootstrap import bootstrap_runtime
from torch_runtime import get_torch

bootstrap_runtime()

def _should_prime_torch_for_gui() -> bool:
    """Return True when the GUI import path should eagerly prime torch."""
    return sys.platform == "win32" and "pytest" not in sys.modules and "PYTEST_CURRENT_TEST" not in os.environ


if _should_prime_torch_for_gui():
    get_torch(context="gui_runtime_bootstrap:prime_torch")
