from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

import torch_runtime


@pytest.fixture(autouse=True)
def _reset_torch_runtime_state():
    original_module = torch_runtime._TORCH_MODULE
    original_error = torch_runtime._TORCH_IMPORT_ERROR
    try:
        torch_runtime._TORCH_MODULE = None
        torch_runtime._TORCH_IMPORT_ERROR = None
        yield
    finally:
        torch_runtime._TORCH_MODULE = original_module
        torch_runtime._TORCH_IMPORT_ERROR = original_error


def test_get_torch_returns_cached_module_without_reimport(monkeypatch):
    cached_module = SimpleNamespace(cuda="stub")
    preload_calls: list[str] = []

    monkeypatch.setattr(torch_runtime, "_TORCH_MODULE", cached_module)
    monkeypatch.setattr(
        torch_runtime,
        "preload_windows_torch_dependencies",
        lambda: preload_calls.append("preload") or [],
    )

    assert torch_runtime.get_torch(context="test") is cached_module
    assert preload_calls == []


def test_get_torch_returns_none_when_import_error_already_cached(monkeypatch):
    preload_calls: list[str] = []

    monkeypatch.setattr(torch_runtime, "_TORCH_IMPORT_ERROR", RuntimeError("cached import failure"))
    monkeypatch.setattr(
        torch_runtime,
        "preload_windows_torch_dependencies",
        lambda: preload_calls.append("preload") or [],
    )

    assert torch_runtime.get_torch(context="test") is None
    assert preload_calls == []


def test_get_torch_imports_and_caches_module(monkeypatch):
    fake_torch = SimpleNamespace(__name__="torch")
    preload_calls: list[str] = []

    monkeypatch.setattr(
        torch_runtime,
        "preload_windows_torch_dependencies",
        lambda: preload_calls.append("preload") or ["dll=ok"],
    )

    real_import = builtins.__import__

    def _fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "torch":
            return fake_torch
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    result = torch_runtime.get_torch(context="test")

    assert result is fake_torch
    assert torch_runtime._TORCH_MODULE is fake_torch
    assert preload_calls == ["preload"]


def test_get_torch_records_import_failure_and_diagnostics(monkeypatch):
    preload_calls: list[str] = []
    diagnostic_calls: list[tuple[str, BaseException, list[str]]] = []
    warning_messages: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(
        torch_runtime,
        "preload_windows_torch_dependencies",
        lambda: preload_calls.append("preload") or ["dll=missing"],
    )
    monkeypatch.setattr(
        torch_runtime,
        "write_startup_diagnostics",
        lambda *, context, error, notes: diagnostic_calls.append((context, error, notes)) or "startup.log",
    )
    monkeypatch.setattr(
        torch_runtime.logger,
        "warning",
        lambda message, *args: warning_messages.append((message, args)),
    )

    real_import = builtins.__import__

    def _fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    assert torch_runtime.get_torch(context="gui") is None
    assert isinstance(torch_runtime.get_torch_import_error(), ImportError)
    assert preload_calls == ["preload"]
    assert diagnostic_calls[0][0] == "gui:import_torch"
    assert str(diagnostic_calls[0][1]) == "torch missing"
    assert diagnostic_calls[0][2] == ["dll=missing"]
    assert warning_messages[0][0] == "PyTorch unavailable; torch-backed features will be limited: %s"
    assert warning_messages[0][1] == (diagnostic_calls[0][1],)
    assert warning_messages[1] == ("Startup diagnostics written to: %s", ("startup.log",))


def test_get_torch_import_error_defaults_to_none():
    assert torch_runtime.get_torch_import_error() is None
