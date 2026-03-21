# YouTube Transcriber

Desktop and CLI transcription tooling for three primary flows:

- YouTube URLs, with caption-first fallback to `yt-dlp` + Whisper
- Local audio/video files, with audio-stream selection and FFmpeg extraction
- Microphone recording in the PyQt6 GUI

## Source Of Truth

- Authored code lives in the repo root Python modules, [`widgets`](./widgets), [`tests`](./tests), [`data`](./data), and the build/launcher scripts.
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

## Packaging

Build the portable Windows bundle with:

```powershell
.\build_standalone.ps1
```

Optional switches:

- `-IncludeCachedModels` to copy cached Hugging Face repos into the bundle
- `-SkipFFmpegBundle` to leave FFmpeg external

`-IncludeCachedModels` only copies models that are both configured and already cached on the build machine running the packaging script.

The PyInstaller spec resolves paths relative to the spec file, not the current working directory, so builds are path-independent.

## Runtime Notes

- `gui_runtime_bootstrap.py` primes `torch` before importing `PyQt6` on Windows to avoid the `c10.dll` / `WinError 1114` startup failure.
- `runtime_bootstrap.py` owns Hugging Face cache env setup. Do not duplicate that logic in launcher scripts or top-level modules.
- Grammar availability checks are intentionally lazy. Startup should report status without eagerly loading GECToR or LanguageTool.
- Lazy grammar status is intentionally honest rather than eager: it may report that GECToR will download or initialize on demand instead of claiming full readiness up front.

## Runtime Paths

- Logs use `app_paths.get_log_path(...)`, not a hard-coded working-directory path.
- `config.json` uses `app_paths.get_config_path()`.
- Hugging Face cache data uses `app_paths.get_model_cache_root()`.
- When the repo or bundle directory is not writable, those paths fall back to a writable app-data location such as `LocalAppData\YouTubeTranscriber`.
