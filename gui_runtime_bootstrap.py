"""GUI-specific runtime bootstrap.

Imports torch before Qt on Windows to avoid DLL-resolution conflicts between
PyTorch and the Qt runtime.
"""

from __future__ import annotations

import sys

from runtime_bootstrap import bootstrap_runtime
from torch_runtime import get_torch

bootstrap_runtime()

if sys.platform == "win32":
    get_torch(context="gui_runtime_bootstrap:prime_torch")
