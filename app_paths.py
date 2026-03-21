"""Helpers for locating writable/runtime paths in source and packaged app modes."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

APP_DIR_NAME = "YouTubeTranscriber"
_WINDOWS_DLL_DIRECTORY_HANDLES: list[object] = []
_REGISTERED_WINDOWS_DLL_DIRS: set[str] = set()


def is_frozen_app() -> bool:
    """Return True when running from a bundled executable."""
    return bool(getattr(sys, "frozen", False))


def get_resource_root() -> Path:
    """Return the directory that contains bundled read-only resources."""
    if is_frozen_app() and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parent


def get_app_root() -> Path:
    """Return the executable/app directory for bundled builds, else the project root."""
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _is_writable_directory(path: Path) -> bool:
    """Best-effort check for whether the app can write to a directory."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def get_writable_app_data_root() -> Path:
    """Return a writable directory for config, logs, temp files, and downloads."""
    candidates = [get_app_root()]

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / APP_DIR_NAME)

    candidates.append(Path.home() / f".{APP_DIR_NAME}")

    for candidate in candidates:
        if _is_writable_directory(candidate):
            return candidate

    return get_app_root()


def get_config_path() -> Path:
    """Return the default config.json path."""
    return get_writable_app_data_root() / "config.json"


def get_log_path(filename: str) -> Path:
    """Return a writable log path for the given filename."""
    return get_writable_app_data_root() / filename


def get_model_cache_root() -> Path:
    """Return the Hugging Face cache root to use for packaged or source runs."""
    bundled_cache = get_app_root() / "hf-cache"
    if bundled_cache.exists() and (_is_writable_directory(bundled_cache) or not is_frozen_app()):
        return bundled_cache
    return get_writable_app_data_root() / "hf-cache"


def get_ffmpeg_search_roots() -> list[Path]:
    """Return high-priority directories to search for bundled FFmpeg binaries."""
    roots: list[Path] = []
    for base in (get_app_root(), get_resource_root()):
        roots.extend((base, base / "ffmpeg", base / "ffmpeg" / "bin", base / "bin"))

    return _dedupe_resolved_paths(roots)


def _dedupe_resolved_paths(paths: list[Path]) -> list[Path]:
    """Return paths in order with duplicates removed after resolution."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def get_runtime_dll_search_roots() -> list[Path]:
    """Return Windows DLL directories that bundled apps should add up front."""
    resource_root = get_resource_root()
    app_root = get_app_root()

    roots = [
        resource_root,
        resource_root / "bin",
        resource_root / "Library" / "bin",
        resource_root / "torch" / "lib",
        resource_root / "ctranslate2",
        app_root,
        app_root / "bin",
    ]

    roots.extend(path for path in resource_root.glob("*.libs") if path.is_dir())

    for nvidia_root in (resource_root / "nvidia", app_root / "nvidia"):
        if nvidia_root.exists():
            roots.extend(path for path in nvidia_root.glob("*/bin") if path.is_dir())

    return _dedupe_resolved_paths(roots)


def _load_ctypes_module():
    """Import ctypes lazily so tests can patch the loader without touching sys.modules."""
    import ctypes

    return ctypes


def register_windows_dll_directory(path: str | Path) -> bool:
    """Register a Windows DLL search directory and retain the returned handle."""
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return False

    resolved_path = str(Path(path).resolve())
    normalized_path = os.path.normcase(resolved_path)
    if normalized_path in _REGISTERED_WINDOWS_DLL_DIRS:
        return True

    try:
        handle = add_dll_directory(resolved_path)
    except OSError:
        return False

    _WINDOWS_DLL_DIRECTORY_HANDLES.append(handle)
    _REGISTERED_WINDOWS_DLL_DIRS.add(normalized_path)
    return True


def _configure_windows_dll_search_paths() -> None:
    """Ensure bundled DLL directories are discoverable on blank Windows installs."""
    existing_roots = [str(path) for path in get_runtime_dll_search_roots() if path.exists()]
    if not existing_roots:
        return

    current_path = os.environ.get("PATH", "")
    path_parts = [part for part in current_path.split(os.pathsep) if part]
    normalized_existing = {os.path.normcase(path) for path in path_parts}
    path_prefix = [path for path in existing_roots if os.path.normcase(path) not in normalized_existing]

    if path_prefix:
        os.environ["PATH"] = os.pathsep.join(path_prefix + path_parts)

    for root in existing_roots:
        register_windows_dll_directory(root)


def _get_torch_dll_dir() -> Path | None:
    """Return the packaged torch DLL directory when available."""
    for base in (get_resource_root(), get_app_root()):
        torch_lib_dir = base / "torch" / "lib"
        if torch_lib_dir.exists():
            return torch_lib_dir.resolve()
    return None


def preload_windows_torch_dependencies() -> list[str]:
    """Preload core torch DLLs from the packaged bundle on Windows.

    Returns a short status list that can be written into startup diagnostics.
    """
    statuses: list[str] = []
    if sys.platform != "win32":
        return statuses

    torch_lib_dir = _get_torch_dll_dir()
    if torch_lib_dir is None:
        statuses.append("torch-lib-dir=missing")
        return statuses

    try:
        ctypes = _load_ctypes_module()
    except Exception as exc:
        statuses.append(f"ctypes-import=failed:{exc}")
        return statuses

    preload_order = (
        "libiomp5md.dll",
        "torch_global_deps.dll",
        "c10.dll",
        "shm.dll",
        "torch_cpu.dll",
        "torch_python.dll",
    )

    for dll_name in preload_order:
        dll_path = torch_lib_dir / dll_name
        if not dll_path.exists():
            statuses.append(f"{dll_name}=missing")
            continue

        try:
            ctypes.WinDLL(str(dll_path))
            statuses.append(f"{dll_name}=ok")
        except OSError as exc:
            statuses.append(f"{dll_name}=failed:{exc.winerror}:{exc}")
            break

    return statuses


def write_startup_diagnostics(*, context: str, error: BaseException, notes: list[str] | None = None) -> Path | None:
    """Write a startup diagnostics file for packaged-app import failures."""
    try:
        diagnostic_path = get_log_path("startup-diagnostics.log")
        lines = [
            f"context={context}",
            f"python={sys.version}",
            f"executable={sys.executable}",
            f"frozen={is_frozen_app()}",
            f"resource_root={get_resource_root()}",
            f"app_root={get_app_root()}",
            f"hf_home={os.environ.get('HF_HOME', '')}",
            f"path={os.environ.get('PATH', '')}",
            "dll_search_roots:",
        ]
        lines.extend(f"  {root}" for root in get_runtime_dll_search_roots())
        if notes:
            lines.append("notes:")
            lines.extend(f"  {note}" for note in notes)
        lines.append("traceback:")
        lines.append("".join(traceback.format_exception(type(error), error, error.__traceback__)))
        diagnostic_path.write_text("\n".join(lines), encoding="utf-8")
        return diagnostic_path
    except Exception:
        return None


def configure_runtime_environment() -> None:
    """Set cache paths and working directory for packaged runs."""
    cache_root = get_model_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "hub")

    if sys.platform == "win32":
        _configure_windows_dll_search_paths()

    if is_frozen_app():
        try:
            os.chdir(get_writable_app_data_root())
        except OSError:
            pass
