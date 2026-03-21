from __future__ import annotations

import os
from pathlib import Path

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
