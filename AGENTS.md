# YouTubeTranscripter Agent Notes

The role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the AGENTS.md file to help prevent future agents from having the same issue. Make sure you always keep this file consistently updated, such that you are not working from deprecated surprises or assumptions from old mistakes or confusion points.

Always commit to PR to remote. Either commit it to an open PR or make a new PR if no PR is open. Make sure local and remote are always synced up. When committing or commenting, always make sure it is concise, straightforward, and understandable. That goes for PRs and commits.

## High-Signal Surprises

- Do not assume this working folder is attached to Git metadata. Verify `.git` exists before planning branch, commit, or PR work.
- On Windows, importing `PyQt6` before `torch` can trigger `WinError 1114` against `torch\\lib\\c10.dll`. The supported startup path is `gui_runtime_bootstrap.py`, which primes `torch` before Qt.
- `runtime_bootstrap.py` owns cache and environment setup for Hugging Face and transformers. Avoid ad hoc `os.environ.setdefault(...)` copies in other modules because tests and packaged runs need the cache root to update deterministically.
- `AppConfig` is the source of truth for functional settings. `QSettings` is only for Qt UI state like splitter geometry.
- Grammar status checks must stay lazy. Startup should not instantiate GECToR or LanguageTool just to render a status label.
- Microphone recording is capture-first batch transcription. Do not add a separate streaming path; after Stop Recording, route the captured audio through the same backend workflow as local files.
- `build/`, `dist/`, `venv/`, `hf-cache/`, logs, `tmp/`, and cache directories are generated artifacts, not source.

## Review Defaults

- Use Auggie before editing unfamiliar files or cross-module behavior.
- Preserve the three primary flows: YouTube URL, local file, and microphone recording.
- Prefer deleting dead code only after confirming there is no live usage via Auggie plus exact search.
- Keep source runs and launcher behavior aligned. If startup or runtime behavior changes in Python, reflect it in `run_gui.bat`.

## Verification

- Default verification loop: `pytest`, `ruff check .`, and `pyright`.
- If you touch launcher behavior, also smoke-test `run_gui.bat` when practical.
