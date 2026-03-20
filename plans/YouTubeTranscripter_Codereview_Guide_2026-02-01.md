# YouTubeTranscripter Codereview Guide

## Purpose

This guide is the handoff for a full cleanup and consolidation review. The reviewer is expected to act like Codex in another session: exhaustive, skeptical, implementation-capable, and careful about false positives.

No skill required.

## Working Mode

- Use Auggie before edits to build cross-file context.
- Treat this as a source-cleanup and runtime-stability review, not a feature sprint.
- Remove overlapping or hanging implementations when they are clearly unused.
- Implement missing pieces only when the intent is clear and the resulting behavior can be verified.
- Keep the authored surface type-safe and verification-clean: `pytest`, `ruff check .`, and `pyright`.

## Authored Surface

- Root modules: `youtube_transcriber.py`, `gui_transcriber.py`, `grammar_postprocessor.py`, `config.py`, `audio_preprocessor.py`, `app_paths.py`, `torch_runtime.py`, `launcher_preflight.py`
- Shared UI/support code: `widgets/`, `themes.py`, `transcript_types.py`
- Packaging/runtime scripts: `run_gui.bat`, `build_standalone.ps1`, `youtube_transcriber.spec`
- Tests: `tests/`
- Data assets: `data/`

Generated artifacts are out of scope except when verifying packaging behavior:

- `build/`
- `dist/`
- `venv/`
- `hf-cache/`
- `tmp/`
- logs and cache directories

## Known Architectural Rules

- `gui_runtime_bootstrap.py` must remain the Windows-safe GUI entry path because `torch` must load before `PyQt6`.
- `runtime_bootstrap.py` is the single place that wires Hugging Face cache environment variables.
- Functional settings belong to `AppConfig`; `QSettings` is only for Qt UI geometry/state.
- Grammar status checks should stay lazy and side-effect free during startup.
- Transcript segment data should move through the system as normalized typed segments from `transcript_types.py`.

## Review Passes

1. Runtime correctness
   - Check startup/bootstrap order, Windows DLL behavior, cache env wiring, FFmpeg/ffprobe discovery, and packaging parity.
2. Core pipeline simplification
   - Look for duplicated Whisper setup, overlapping fallback logic, stale config snapshots, and inconsistent exception contracts.
3. GUI thread safety
   - Confirm worker threads only use snapshotted config/state, recording shutdown is deterministic, and queue messages match the UI handlers.
4. Grammar backends
   - Keep startup lazy, verify GECToR/LanguageTool fallback honesty, and delete unused or redundant backend glue.
5. Packaging and launcher hygiene
   - Keep `run_gui.bat`, `launcher_preflight.py`, `build_standalone.ps1`, and `youtube_transcriber.spec` aligned with the actual Python runtime path.
6. Dead-code sweep
   - Use Auggie plus exact search before removing widgets, theme helpers, scripts, or packaging helpers.
7. Docs and repo hygiene
   - Update `README.md`, `AGENTS.md`, and any stale plan docs when new surprises are discovered.

## Immediate Hotspots

- `gui_transcriber.py`
  - Background workers should never read live widget state after they start.
  - Queue message names must match `process_queue()` handlers.
  - Recording shutdown and post-record transcription ownership should stay explicit.
- `youtube_transcriber.py`
  - Shared transcript normalization and typed segment handling should stay aligned across YouTube, local-file, and recording flows.
  - Fallbacks must preserve preprocessing inputs and not silently change semantics.
- `grammar_postprocessor.py`
  - Accept legacy dict-shaped segments at the public boundary, but normalize them immediately.
  - Keep runtime probes non-eager.
- `run_gui.bat`
  - Dependency probes must validate the real startup path, not a synthetic import order that reintroduces the Windows torch/Qt crash.
- `youtube_transcriber.spec`
  - Paths should resolve relative to the spec file, not `Path.cwd()`.

## Acceptance Criteria

- `pytest` passes
- `ruff check .` passes
- `pyright` passes
- Source GUI launch path no longer reproduces the `PyQt6` before `torch` `WinError 1114` failure
- Docs accurately describe the real repo boundaries and runtime behavior
