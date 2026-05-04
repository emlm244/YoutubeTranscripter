# Speech Transcriber

Desktop and CLI transcription tooling for three primary flows:

- YouTube URLs, with caption-first fallback to `yt-dlp` plus OpenAI or local Whisper
- Local audio/video files, with audio-stream selection, FFmpeg extraction, and selectable OpenAI/local backends
- Microphone recording in the PyQt6 GUI, transcribed after capture with the selected batch backend

## Source Of Truth

- Authored code lives in the repo root Python modules, [`widgets`](./widgets), [`tests`](./tests), [`data`](./data), and the launcher scripts.
- Generated state is disposable: `build/`, `dist/`, `venv/`, `hf-cache/`, logs, caches, and `tmp/`.
- Functional app settings live in `AppConfig`; Qt `QSettings` is only for window/splitter state.

## Setup

Windows source run:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
run_gui.bat
```

The launcher now follows the same runtime bootstrap path as the GUI and will not fail just because PyTorch is unavailable. Whisper can still run on the CTranslate2 CUDA backend or CPU fallback.
Normal launches now skip optional preflight/GPU diagnostics so the window opens faster. Set `YT_VERBOSE_STARTUP=1` before running [`run_gui.bat`](./run_gui.bat) when you want those extra launcher diagnostics back.

OpenAI-backed transcription requires `OPENAI_API_KEY` in the environment. If an API key was ever pasted into chat, logs, or source, revoke it in the OpenAI dashboard and create a fresh key before launching the app.

```powershell
$env:OPENAI_API_KEY = "<new-openai-api-key>"
run_gui.bat
```

## GUI

Source launcher:

```powershell
run_gui.bat
```

Direct Python entrypoint:

```powershell
python gui_transcriber.py
```

The GUI keeps functional settings in `AppConfig` and uses Qt `QSettings` only for window and splitter state.
Microphone enumeration and dependency/GPU diagnostics are deferred until after the window paints so startup stays responsive.
Batch transcription can run with OpenAI, local Whisper, or Compare mode. Compare mode runs OpenAI first and local Whisper second, then shows both transcripts with labels for benchmarking. The microphone card records locally first, then sends the captured audio through the same backend selection as local files.

## CLI

The repo still ships a CLI entrypoint in [`youtube_transcriber.py`](./youtube_transcriber.py).

Basic usage:

```powershell
python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ output.txt
python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --format timestamped
python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --verbose
```

Notes:

- The CLI currently targets the YouTube URL flow.
- `save_transcript()` keeps outputs inside the current working directory tree and sanitizes the final filename for Windows-safe writes.
- The YouTube fallback helpers intentionally preserve their older best-effort contracts: `download_audio()` returns `None` on failure, `transcribe_audio()` returns `(None, None)`, and `transcribe_local_file()` raises typed exceptions.

## Development Checks

```powershell
.\venv\Scripts\activate
pytest
ruff check .
pyright
```

`pyproject.toml` scopes Ruff and Pyright to the authored surface so generated bundle contents do not pollute the verification loop.

## Runtime Notes

- `gui_runtime_bootstrap.py` primes `torch` before importing `PyQt6` on Windows to avoid the `c10.dll` / `WinError 1114` startup failure.
- `runtime_bootstrap.py` owns Hugging Face cache env setup. Do not duplicate that logic in launcher scripts or top-level modules.
- Grammar availability checks are intentionally lazy. Startup should report status without eagerly loading GECToR or LanguageTool.
- Lazy grammar status is intentionally honest rather than eager: it may report that GECToR will download or initialize on demand instead of claiming full readiness up front.
- `launcher_preflight.py` is now an on-demand diagnostic helper rather than a mandatory launch step.

## Runtime Paths

- Logs use `app_paths.get_log_path(...)`, not a hard-coded working-directory path.
- `config.json` uses `app_paths.get_config_path()`.
- Hugging Face cache data uses `app_paths.get_model_cache_root()`.
- When the repo directory is not writable, those paths fall back to a writable app-data location such as `LocalAppData\YouTubeTranscriber`, the legacy directory name preserved for backward compatibility.
