from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import app_paths


@pytest.fixture(autouse=True)
def _restore_runtime_environment():
    original_values = {
        key: os.environ.get(key)
        for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "PATH")
    }
    original_handles = list(app_paths._WINDOWS_DLL_DIRECTORY_HANDLES)
    original_registered = set(app_paths._REGISTERED_WINDOWS_DLL_DIRS)
    try:
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        app_paths._WINDOWS_DLL_DIRECTORY_HANDLES[:] = original_handles
        app_paths._REGISTERED_WINDOWS_DLL_DIRS.clear()
        app_paths._REGISTERED_WINDOWS_DLL_DIRS.update(original_registered)


def test_get_ffmpeg_search_roots_includes_app_and_resource_locations(monkeypatch, tmp_path: Path):
    app_root = tmp_path / "app-root"
    resource_root = tmp_path / "resource-root"
    app_root.mkdir()
    resource_root.mkdir()

    monkeypatch.setattr(app_paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(app_paths, "get_resource_root", lambda: resource_root)

    roots = app_paths.get_ffmpeg_search_roots()

    assert app_root.resolve() in roots
    assert (app_root / "ffmpeg").resolve() in roots
    assert (app_root / "ffmpeg" / "bin").resolve() in roots
    assert resource_root.resolve() in roots
    assert (resource_root / "bin").resolve() in roots


def test_get_model_cache_root_prefers_bundled_cache(monkeypatch, tmp_path: Path):
    bundled_root = tmp_path / "bundle"
    bundled_cache = bundled_root / "hf-cache"
    bundled_cache.mkdir(parents=True)

    monkeypatch.setattr(app_paths, "get_app_root", lambda: bundled_root)
    monkeypatch.setattr(app_paths, "get_writable_app_data_root", lambda: tmp_path / "writable")

    assert app_paths.get_model_cache_root() == bundled_cache


def test_get_runtime_dll_search_roots_includes_packaged_dependency_dirs(monkeypatch, tmp_path: Path):
    app_root = tmp_path / "app-root"
    resource_root = tmp_path / "resource-root"
    (resource_root / "torch" / "lib").mkdir(parents=True)
    (resource_root / "ctranslate2").mkdir(parents=True)
    (resource_root / "numpy.libs").mkdir()
    (resource_root / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (resource_root / "nvidia" / "cuda_runtime" / "bin").mkdir(parents=True)
    app_root.mkdir()

    monkeypatch.setattr(app_paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(app_paths, "get_resource_root", lambda: resource_root)

    roots = app_paths.get_runtime_dll_search_roots()

    assert resource_root.resolve() in roots
    assert (resource_root / "torch" / "lib").resolve() in roots
    assert (resource_root / "ctranslate2").resolve() in roots
    assert (resource_root / "numpy.libs").resolve() in roots
    assert (resource_root / "nvidia" / "cublas" / "bin").resolve() in roots
    assert (resource_root / "nvidia" / "cuda_runtime" / "bin").resolve() in roots
    assert app_root.resolve() in roots


def test_configure_runtime_environment_prefixes_existing_windows_dll_roots(monkeypatch, tmp_path: Path):
    cache_root = tmp_path / "cache"
    app_root = tmp_path / "app-root"
    resource_root = tmp_path / "resource-root"
    (resource_root / "torch" / "lib").mkdir(parents=True)
    app_root.mkdir()

    calls: list[str] = []

    monkeypatch.setattr(app_paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(app_paths, "get_model_cache_root", lambda: cache_root)
    monkeypatch.setattr(app_paths, "get_resource_root", lambda: resource_root)
    monkeypatch.setattr(app_paths.sys, "platform", "win32")
    monkeypatch.setattr(app_paths.os, "add_dll_directory", lambda path: calls.append(path), raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join(["existing-bin"]))

    app_paths.configure_runtime_environment()

    assert cache_root.is_dir()
    assert os.environ["HF_HOME"] == str(cache_root)
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(cache_root / "hub")
    assert str(resource_root.resolve()) in calls
    assert str((resource_root / "torch" / "lib").resolve()) in calls
    assert str(app_root.resolve()) in calls
    assert os.environ["PATH"].split(os.pathsep)[0] == str(resource_root.resolve())


def test_register_windows_dll_directory_retains_handle(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    handles = []

    def _fake_add_dll_directory(path: str):
        calls.append(path)
        handle = object()
        handles.append(handle)
        return handle

    dll_dir = tmp_path / "dlls"
    dll_dir.mkdir()

    monkeypatch.setattr(app_paths.os, "add_dll_directory", _fake_add_dll_directory, raising=False)
    app_paths._WINDOWS_DLL_DIRECTORY_HANDLES.clear()
    app_paths._REGISTERED_WINDOWS_DLL_DIRS.clear()

    assert app_paths.register_windows_dll_directory(dll_dir) is True
    assert app_paths.register_windows_dll_directory(dll_dir) is True
    assert calls == [str(dll_dir.resolve())]
    assert app_paths._WINDOWS_DLL_DIRECTORY_HANDLES == handles


def test_resource_and_app_roots_switch_for_frozen_build(monkeypatch, tmp_path: Path):
    resource_root = tmp_path / "bundle"
    executable = tmp_path / "dist" / "app.exe"
    resource_root.mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")

    monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app_paths.sys, "_MEIPASS", str(resource_root), raising=False)
    monkeypatch.setattr(app_paths.sys, "executable", str(executable))

    assert app_paths.is_frozen_app() is True
    assert app_paths.get_resource_root() == resource_root.resolve()
    assert app_paths.get_app_root() == executable.parent.resolve()


def test_get_writable_app_data_root_prefers_localappdata_when_app_root_not_writable(monkeypatch, tmp_path: Path):
    app_root = tmp_path / "app-root"
    local_app_data = tmp_path / "local"
    home_dir = tmp_path / "home"
    app_root.mkdir()
    local_app_data.mkdir()
    home_dir.mkdir()

    monkeypatch.setattr(app_paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(app_paths.Path, "home", classmethod(lambda cls: home_dir))
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    writable_calls: list[Path] = []

    def _fake_is_writable(path: Path) -> bool:
        writable_calls.append(path)
        return path == local_app_data / app_paths.APP_DIR_NAME

    monkeypatch.setattr(app_paths, "_is_writable_directory", _fake_is_writable)

    assert app_paths.get_writable_app_data_root() == (local_app_data / app_paths.APP_DIR_NAME)
    assert writable_calls[0] == app_root


def test_get_writable_app_data_root_falls_back_to_home(monkeypatch, tmp_path: Path):
    app_root = tmp_path / "app-root"
    home_dir = tmp_path / "home"
    app_root.mkdir()
    home_dir.mkdir()

    monkeypatch.setattr(app_paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(app_paths.Path, "home", classmethod(lambda cls: home_dir))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(app_paths, "_is_writable_directory", lambda path: path == home_dir / f".{app_paths.APP_DIR_NAME}")

    assert app_paths.get_writable_app_data_root() == (home_dir / f".{app_paths.APP_DIR_NAME}")


def test_get_config_and_log_paths_use_writable_root(monkeypatch, tmp_path: Path):
    writable_root = tmp_path / "data-root"
    monkeypatch.setattr(app_paths, "get_writable_app_data_root", lambda: writable_root)

    assert app_paths.get_config_path() == writable_root / "config.json"
    assert app_paths.get_log_path("gui.log") == writable_root / "gui.log"


def test_get_model_cache_root_falls_back_when_bundled_cache_missing(monkeypatch, tmp_path: Path):
    app_root = tmp_path / "bundle"
    writable_root = tmp_path / "writable"
    app_root.mkdir()
    writable_root.mkdir()

    monkeypatch.setattr(app_paths, "get_app_root", lambda: app_root)
    monkeypatch.setattr(app_paths, "get_writable_app_data_root", lambda: writable_root)

    assert app_paths.get_model_cache_root() == writable_root / "hf-cache"


def test_dedupe_resolved_paths_preserves_order(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    paths = [base, base / ".", base.resolve()]

    result = app_paths._dedupe_resolved_paths(paths)

    assert result == [base.resolve()]


def test_register_windows_dll_directory_handles_missing_api(monkeypatch, tmp_path: Path):
    monkeypatch.delattr(app_paths.os, "add_dll_directory", raising=False)
    assert app_paths.register_windows_dll_directory(tmp_path) is False


def test_register_windows_dll_directory_handles_oserror(monkeypatch, tmp_path: Path):
    dll_dir = tmp_path / "dlls"
    dll_dir.mkdir()
    monkeypatch.setattr(app_paths.os, "add_dll_directory", lambda path: (_ for _ in ()).throw(OSError("boom")), raising=False)
    assert app_paths.register_windows_dll_directory(dll_dir) is False


def test_configure_windows_dll_search_paths_skips_when_no_existing_roots(monkeypatch):
    monkeypatch.setattr(app_paths, "get_runtime_dll_search_roots", lambda: [])
    monkeypatch.setenv("PATH", "existing")

    app_paths._configure_windows_dll_search_paths()

    assert os.environ["PATH"] == "existing"


def test_get_torch_dll_dir_prefers_resource_root(monkeypatch, tmp_path: Path):
    resource_root = tmp_path / "resource"
    app_root = tmp_path / "app"
    (resource_root / "torch" / "lib").mkdir(parents=True)
    app_root.mkdir()

    monkeypatch.setattr(app_paths, "get_resource_root", lambda: resource_root)
    monkeypatch.setattr(app_paths, "get_app_root", lambda: app_root)

    assert app_paths._get_torch_dll_dir() == (resource_root / "torch" / "lib").resolve()


def test_preload_windows_torch_dependencies_reports_missing_dir(monkeypatch):
    monkeypatch.setattr(app_paths.sys, "platform", "win32")
    monkeypatch.setattr(app_paths, "_get_torch_dll_dir", lambda: None)

    assert app_paths.preload_windows_torch_dependencies() == ["torch-lib-dir=missing"]


def test_preload_windows_torch_dependencies_reports_ctypes_failure(monkeypatch, tmp_path: Path):
    torch_lib_dir = tmp_path / "torch" / "lib"
    torch_lib_dir.mkdir(parents=True)
    monkeypatch.setattr(app_paths.sys, "platform", "win32")
    monkeypatch.setattr(app_paths, "_get_torch_dll_dir", lambda: torch_lib_dir)
    monkeypatch.setattr(app_paths, "_load_ctypes_module", lambda: (_ for _ in ()).throw(RuntimeError("ctypes boom")))

    statuses = app_paths.preload_windows_torch_dependencies()

    assert statuses == ["ctypes-import=failed:ctypes boom"]


def test_preload_windows_torch_dependencies_reports_missing_and_failed_dlls(monkeypatch, tmp_path: Path):
    torch_lib_dir = tmp_path / "torch" / "lib"
    torch_lib_dir.mkdir(parents=True)
    first = torch_lib_dir / "libiomp5md.dll"
    second = torch_lib_dir / "torch_global_deps.dll"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    class _FakeOSError(OSError):
        def __init__(self):
            super().__init__("load failed")
            self.winerror = 1114

    def _fake_win_dll(path: str):
        if path.endswith("torch_global_deps.dll"):
            raise _FakeOSError()
        return object()

    monkeypatch.setattr(app_paths.sys, "platform", "win32")
    monkeypatch.setattr(app_paths, "_get_torch_dll_dir", lambda: torch_lib_dir)
    monkeypatch.setattr(app_paths, "_load_ctypes_module", lambda: SimpleNamespace(WinDLL=_fake_win_dll))

    statuses = app_paths.preload_windows_torch_dependencies()

    assert statuses[0] == "libiomp5md.dll=ok"
    assert statuses[1].startswith("torch_global_deps.dll=failed:1114:")


def test_write_startup_diagnostics_writes_context_and_notes(monkeypatch, tmp_path: Path):
    diagnostic_path = tmp_path / "startup-diagnostics.log"
    monkeypatch.setattr(app_paths, "get_log_path", lambda filename: diagnostic_path)
    monkeypatch.setattr(app_paths, "get_runtime_dll_search_roots", lambda: [tmp_path / "dll-a", tmp_path / "dll-b"])

    result = app_paths.write_startup_diagnostics(
        context="torch",
        error=RuntimeError("boom"),
        notes=["note-a", "note-b"],
    )

    assert result == diagnostic_path
    text = diagnostic_path.read_text(encoding="utf-8")
    assert "context=torch" in text
    assert "notes:" in text
    assert "note-a" in text
    assert "RuntimeError: boom" in text


def test_write_startup_diagnostics_returns_none_on_failure(monkeypatch, tmp_path: Path):
    broken_path = tmp_path / "missing" / "startup-diagnostics.log"
    monkeypatch.setattr(app_paths, "get_log_path", lambda filename: broken_path)

    assert app_paths.write_startup_diagnostics(context="x", error=RuntimeError("boom")) is None


def test_configure_runtime_environment_changes_directory_for_frozen_app(monkeypatch, tmp_path: Path):
    cache_root = tmp_path / "cache"
    writable_root = tmp_path / "writable"
    writable_root.mkdir()
    cwd_calls: list[Path] = []

    monkeypatch.setattr(app_paths, "get_model_cache_root", lambda: cache_root)
    monkeypatch.setattr(app_paths, "get_writable_app_data_root", lambda: writable_root)
    monkeypatch.setattr(app_paths.sys, "platform", "linux")
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: True)
    monkeypatch.setattr(app_paths.os, "chdir", lambda path: cwd_calls.append(Path(path)))

    app_paths.configure_runtime_environment()

    assert os.environ["HF_HOME"] == str(cache_root)
    assert cwd_calls == [writable_root]


def test_configure_runtime_environment_ignores_chdir_failure(monkeypatch, tmp_path: Path):
    cache_root = tmp_path / "cache"
    writable_root = tmp_path / "writable"
    monkeypatch.setattr(app_paths, "get_model_cache_root", lambda: cache_root)
    monkeypatch.setattr(app_paths, "get_writable_app_data_root", lambda: writable_root)
    monkeypatch.setattr(app_paths.sys, "platform", "linux")
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: True)
    monkeypatch.setattr(app_paths.os, "chdir", lambda path: (_ for _ in ()).throw(OSError("nope")))

    app_paths.configure_runtime_environment()

    assert os.environ["HF_HOME"] == str(cache_root)
