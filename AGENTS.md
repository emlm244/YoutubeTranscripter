# YouTubeTranscripter Agent Notes

This file records repo-specific surprises and review guidance for future Codex sessions.

## High-Signal Surprises

- Do not assume this working folder is attached to Git metadata. Verify `.git` exists before planning branch, commit, or PR work.
- On Windows, importing `PyQt6` before `torch` can trigger `WinError 1114` against `torch\\lib\\c10.dll`. The supported startup path is `gui_runtime_bootstrap.py`, which primes `torch` before Qt.
- `runtime_bootstrap.py` owns cache/env setup for Hugging Face and transformers. Avoid ad hoc `os.environ.setdefault(...)` copies in other modules because tests and packaged runs need the cache root to update deterministically.
- `AppConfig` is the source of truth for functional settings. `QSettings` is only for Qt UI state like splitter geometry.
- Grammar status checks must stay lazy. Startup should not instantiate GECToR or LanguageTool just to render a status label.
- `build/`, `dist/`, `venv/`, `hf-cache/`, logs, `tmp/`, and cache directories are generated artifacts, not source.

## Review Defaults

- Use Auggie before editing unfamiliar files or cross-module behavior.
- Preserve the three primary flows: YouTube URL, local file, and microphone recording.
- Prefer deleting dead code only after confirming there is no live usage via Auggie plus exact search.
- Keep source runs and packaged runs behaviorally aligned. If startup/runtime behavior changes in Python, reflect it in `run_gui.bat` and packaging scripts.

## Verification

- Default verification loop:
  - `pytest`
  - `ruff check .`
  - `pyright`
- If you touch packaging or launcher behavior, also smoke-test `run_gui.bat` and `build_standalone.ps1` when practical.
