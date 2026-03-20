from __future__ import annotations

import os
from pathlib import Path

import app_paths


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


def test_preload_windows_torch_dependencies_loads_core_dlls(monkeypatch, tmp_path: Path):
    torch_lib_dir = tmp_path / "torch" / "lib"
    torch_lib_dir.mkdir(parents=True)
    for name in ("libiomp5md.dll", "torch_global_deps.dll", "c10.dll", "shm.dll", "torch_cpu.dll", "torch_python.dll"):
        (torch_lib_dir / name).write_bytes(b"")

    loaded: list[str] = []

    class _FakeCtypes:
        @staticmethod
        def WinDLL(path: str):
            loaded.append(path)

    monkeypatch.setattr(app_paths, "_get_torch_dll_dir", lambda: torch_lib_dir)
    monkeypatch.setattr(app_paths.sys, "platform", "win32")
    monkeypatch.setitem(__import__("sys").modules, "ctypes", _FakeCtypes())

    statuses = app_paths.preload_windows_torch_dependencies()

    assert statuses == [
        "libiomp5md.dll=ok",
        "torch_global_deps.dll=ok",
        "c10.dll=ok",
        "shm.dll=ok",
        "torch_cpu.dll=ok",
        "torch_python.dll=ok",
    ]
    assert loaded == [str(torch_lib_dir / name) for name in (
        "libiomp5md.dll",
        "torch_global_deps.dll",
        "c10.dll",
        "shm.dll",
        "torch_cpu.dll",
        "torch_python.dll",
    )]


def test_preload_windows_torch_dependencies_reports_failures(monkeypatch, tmp_path: Path):
    torch_lib_dir = tmp_path / "torch" / "lib"
    torch_lib_dir.mkdir(parents=True)
    (torch_lib_dir / "c10.dll").write_bytes(b"")

    class _FakeOSError(OSError):
        def __init__(self):
            super().__init__("boom")
            self.winerror = 1114

    class _FakeCtypes:
        @staticmethod
        def WinDLL(path: str):
            raise _FakeOSError()

    monkeypatch.setattr(app_paths, "_get_torch_dll_dir", lambda: torch_lib_dir)
    monkeypatch.setattr(app_paths.sys, "platform", "win32")
    monkeypatch.setitem(__import__("sys").modules, "ctypes", _FakeCtypes())

    statuses = app_paths.preload_windows_torch_dependencies()

    assert statuses[0] == "libiomp5md.dll=missing"
    assert statuses[1] == "torch_global_deps.dll=missing"
    assert statuses[2].startswith("c10.dll=failed:1114:")
